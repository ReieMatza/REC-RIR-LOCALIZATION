# @author: Changsheng Quan
# @email: quanchangsheng@westlake.edu.cn
# LastEditors: Pengyu Wang
# @description: code for generating RIR pairs.


import os

os.environ["OMP_NUM_THREADS"] = str(
    1
)  

import json
from typing import *
from jsonargparse import ArgumentParser
import numpy as np
import tqdm
from numpy.linalg import norm
from numpy.random import uniform
import multiprocessing
from functools import partial
import warnings
import inspect
from scipy.optimize import minimize
from pathlib import Path
import soundfile as sf

import gpuRIR


gpuRIR.activateMixedPrecision(False)
gpuRIR.activateLUT(False)


def compute_angle_radius(pos_src: np.ndarray, pos_rcv: np.ndarray, room_sz: Optional[List[float]] = None) -> Tuple[float, float]:
    """
    Compute angle and radius from microphone array center to source.
    
    Angle is measured in global room coordinates:
    - 0° = positive x-axis
    - 90° = positive y-axis
    - 180° = negative x-axis
    - 270° = negative y-axis
    
    If room_sz is provided (when using half-space restriction), angles > 180° are
    reflected to [0, 180] range (e.g., 270° becomes 90°, 350° becomes 10°).
    This gives an unsigned angular deviation, treating left and right symmetrically.
    """
    src = np.array(pos_src)[0]
    rcv = np.array(pos_rcv)
    rcv_center = rcv.mean(axis=0)
    delta = src - rcv_center
    radius_m = float(np.linalg.norm(delta[:2]))
    
    # Compute angle in standard global coordinates [0, 360)
    # 0° = +x, 90° = +y, 180° = -x, 270° = -y
    angle_deg = np.degrees(np.arctan2(delta[1], delta[0])) % 360.0
    
    if room_sz is not None:
        # Limit to [0, 180] by reflecting angles > 180
        if angle_deg > 180.0:
            angle_deg = 360.0 - angle_deg
    
    return angle_deg, radius_m


def save_config_to_file(args, file_path):
    with open(file_path, "w") as json_file:
        json.dump(args.__dict__, json_file, indent=4)


def estimate_minimal_RT60(room_sz: Union[List[float], np.ndarray]) -> float:
    V = 1.0
    for v in room_sz:
        V = V * v
    S = (
        room_sz[0] * room_sz[1] + room_sz[0] * room_sz[2] + room_sz[1] * room_sz[2]
    ) * 2
    RT60 = 0.161 * V / S
    return RT60


def is_valid_RT60_for_room(
    room_sz: Union[List[float], np.ndarray], RT60: float, eps: float = 1e-4
) -> bool:
    RT60m = estimate_minimal_RT60(room_sz)
    if RT60 < RT60m + eps:
        return False
    else:
        return True


def is_valid_beta(beta: Union[List[float], np.ndarray]) -> bool:
    return not np.isclose(beta, 0).any()


def beta_SabineEstimation(room_sz, T60, abs_weights=[1.0] * 6):
    """Estimation of the reflection coefficients needed to have the desired reverberation time. (The code is taken from gpuRIR)

    Parameters
    ----------
    room_sz : 3 elements list or numpy array
            Size of the room (in meters).
    T60 : float
            Reverberation time of the room (seconds to reach 60dB attenuation).
    abs_weights : array_like with 6 elements, optional
            Absoprtion coefficient ratios of the walls (the default is [1.0]*6).

    Returns
    -------
    ndarray with 6 elements
            Reflection coefficients of the walls as $[beta_{x0}, beta_{x1}, beta_{y0}, beta_{y1}, beta_{z0}, beta_{z1}]$,
            where $beta_{x0}$ is the coeffcient of the wall parallel to the x axis closest
            to the origin of coordinates system and $beta_{x1}$ the farthest.
    """

    def t60error(x, T60, room_sz, abs_weights):
        alpha = x * abs_weights
        Sa = (
            (alpha[0] + alpha[1]) * room_sz[1] * room_sz[2]
            + (alpha[2] + alpha[3]) * room_sz[0] * room_sz[2]
            + (alpha[4] + alpha[5]) * room_sz[0] * room_sz[1]
        )
        V = np.prod(room_sz)
        if Sa == 0:
            return T60 - 0  # Anechoic chamber
        return abs(T60 - 0.161 * V / Sa)  # Sabine's formula

    abs_weights /= np.array(abs_weights).max()
    result = minimize(t60error, 0.5, args=(T60, room_sz, abs_weights), bounds=[[0, 1]])
    return np.sqrt(1 - result.x * abs_weights).astype(np.float32), result.fun


def generate_rir_cpu(
    room_sz: Union[List[float], np.ndarray],
    pos_src: Union[List[List[float]], np.ndarray],
    pos_rcv: Union[List[List[float]], np.ndarray],
    RT60: float,
    fs: int,
    beta: Optional[np.ndarray] = None,
    sensor_orientations=None,
    sensor_directivity=None,
    sound_velocity: float = 343,
):
    if len(pos_src) == 0:
        return None

    assert RT60 >= 0, RT60
    filter_length: int = int((RT60 + 0.1) * fs)

    room_sz = np.array(room_sz)
    pos_src = np.array(pos_src)
    pos_rcv = np.array(pos_rcv)

    if np.ndim(pos_src) == 1:
        pos_src = np.reshape(pos_src, (1, -1))
    if np.ndim(room_sz) == 1:
        room_sz = np.reshape(room_sz, (1, -1))
    if np.ndim(pos_rcv) == 1:
        pos_rcv = np.reshape(pos_rcv, (1, -1))

    assert room_sz.shape == (1, 3)
    assert pos_src.shape[1] == 3
    assert pos_rcv.shape[1] == 3

    n_src = pos_src.shape[0]
    n_mic = pos_rcv.shape[0]

    if sensor_orientations is None:
        sensor_orientations = np.zeros((2, n_src))
    else:
        raise NotImplementedError(sensor_orientations)

    if sensor_directivity is None:
        sensor_directivity = "omnidirectional"
    else:
        raise NotImplementedError(sensor_directivity)

    assert filter_length is not None
    rir = np.zeros((n_src, n_mic, filter_length), dtype=np.float64)
    import rir_generator

    for k in range(n_src):
        temp = rir_generator.generate(
            c=sound_velocity,
            fs=fs,
            r=np.ascontiguousarray(pos_rcv),
            s=np.ascontiguousarray(pos_src[k, :]),
            L=np.ascontiguousarray(room_sz[0, :]),
            beta=beta,
            reverberation_time=RT60,
            nsample=filter_length,
            mtype=rir_generator.mtype.omnidirectional,
        )
        rir[k, :, :] = np.asarray(temp.T)

    assert rir.shape[0] == n_src
    assert rir.shape[1] == n_mic
    assert rir.shape[2] == filter_length

    assert not np.any(
        np.isnan(rir)
    ), f"{np.sum(np.isnan(rir))} values of {rir.size} are NaN."
    return rir


def generate_rir_gpu(
    room_sz: Union[List[float], np.ndarray],
    pos_src: Union[List[List[float]], np.ndarray],
    pos_rcv: Union[List[List[float]], np.ndarray],
    RT60: float,
    fs: int,
    sound_velocity: float = 343,
    att_diff: float = None,
    beta: Optional[np.ndarray] = None,
):
    if len(pos_src) == 0:
        return None

    if RT60 == 0:  # direct-path rir
        RT60 = 1  # 随便给一个值，防止出错
        Tmax = 0.1
        nb_img = [1, 1, 1]
        beta = [0] * 6
        Tdiff = None
    else:
        Tmax = gpuRIR.att2t_SabineEstimator(60.0, RT60)
        if att_diff is not None:
            # print(att_diff,RT60)
            Tdiff = gpuRIR.att2t_SabineEstimator(att_diff, RT60)
            nb_img = gpuRIR.t2n(Tdiff, room_sz)
        else:
            Tdiff = None
            nb_img = gpuRIR.t2n(Tmax, room_sz)

        if beta is None:
            beta = gpuRIR.beta_SabineEstimation(
                room_sz, RT60
            )  # reflection coefficients

            if is_valid_beta(beta) == False:
                warnings.warn(
                    f"beta is invalid for gpuRIR, which might indicate the given RT60={RT60} could not achieved with the given room_sz={room_sz}"
                )

    rir = gpuRIR.simulateRIR(
        room_sz=room_sz,
        beta=beta,
        pos_src=pos_src,
        pos_rcv=pos_rcv,
        nb_img=nb_img,
        Tmax=Tmax,
        Tdiff=Tdiff,
        fs=fs,
        c=sound_velocity,
    )

    return rir


def normalize(vec: np.ndarray) -> np.ndarray:
    # get unit vector
    vec = vec / norm(vec)
    vec = vec / norm(vec)
    assert np.isclose(norm(vec), 1), "norm of vec is not close to 1"
    return vec


def _rmax_at_angle(
    cx: float, cy: float,
    Lx: float, Ly: float,
    theta_rad: float,
    min_dist: float,
) -> float:
    """Maximum radius from mic center (cx, cy) in direction theta_rad before
    hitting a wall, respecting the min_dist clearance on each wall.

    Returns 0.0 when the mic is already closer than min_dist to every wall in
    that direction (degenerate room / mic placement).
    """
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    limits = []
    if cos_t > 1e-12:
        limits.append((Lx - min_dist - cx) / cos_t)
    elif cos_t < -1e-12:
        limits.append((cx - min_dist) / (-cos_t))
    if sin_t > 1e-12:
        limits.append((Ly - min_dist - cy) / sin_t)
    elif sin_t < -1e-12:
        limits.append((cy - min_dist) / (-sin_t))
    valid = [v for v in limits if v > 0]
    return float(min(valid)) if valid else 0.0


def plot_room(
    room_sz: Union[List[float], np.ndarray],
    pos_src: np.ndarray,
    pos_rcv: np.ndarray,
    pos_noise: np.ndarray,
    saveto: str = None,
) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图

    plt.close("all")
    fig = plt.figure(figsize=(10, 8))
    ax = Axes3D(fig)
    fig.add_axes(ax)
    ax.scatter(pos_rcv[:, 0], pos_rcv[:, 1], pos_rcv[:, 2])
    if len(pos_rcv) > 2:
        # draw the first half mics with different color for checking the rotation
        ax.scatter(
            pos_rcv[: len(pos_rcv) // 2, 0],
            pos_rcv[: len(pos_rcv) // 2, 1],
            pos_rcv[: len(pos_rcv) // 2, 2],
            c="r",
        )
    ax.scatter(pos_src[:, 0], pos_src[:, 1], pos_src[:, 2])
    if pos_noise is not None and len(pos_noise) > 0:
        ax.scatter(pos_noise[:, 0], pos_noise[:, 1], pos_noise[:, 2])

    ax.set(xlabel="X", ylabel="Y", zlabel="Z")
    ax.set_xlim3d([0, room_sz[0]])
    ax.set_ylim3d([0, room_sz[1]])
    ax.set_zlim3d([0, room_sz[2]])
    plt.show(block=True)
    if saveto is not None:
        plt.savefig("config/rirs/images/" + saveto + ".jpg")
    plt.close()


def circular_array_geometry(radius: float, mic_num: int) -> np.ndarray:
    # 生成圆阵的拓扑（原点为中心），后期可以通过旋转、改变中心的位置来实现阵列位置的改变
    pos_rcv = np.empty((mic_num, 3))
    v1 = np.array([1, 0, 0])  # 第一个麦克风的位置（要求单位向量）
    v1 = normalize(v1)  # 单位向量
    # 将v1绕原点水平旋转angle角度，来生成其他mic的位置
    angles = np.arange(0, 2 * np.pi, 2 * np.pi / mic_num)
    for idx, angle in enumerate(angles):
        x = v1[0] * np.cos(angle) - v1[1] * np.sin(angle)
        y = v1[0] * np.sin(angle) + v1[1] * np.cos(angle)
        pos_rcv[idx, :] = normalize(np.array([x, y, 0]))
    # 设置radius
    pos_rcv *= radius
    return pos_rcv


def circular_with_center_array_geometry(radius: float, mic_num: int) -> np.ndarray:
    # 生成圆阵的拓扑（原点为中心），后期可以通过旋转、改变中心的位置来实现阵列位置的改变
    pos_rcv = np.empty((mic_num, 3))
    v1 = np.array([1, 0, 0])  # 第一个麦克风的位置（要求单位向量）
    v1 = normalize(v1)  # 单位向量
    # 将v1绕原点水平旋转angle角度，来生成其他mic的位置
    angles = np.arange(0, 2 * np.pi, 2 * np.pi / mic_num)
    for idx, angle in enumerate(angles):
        x = v1[0] * np.cos(angle) - v1[1] * np.sin(angle)
        y = v1[0] * np.sin(angle) + v1[1] * np.cos(angle)
        pos_rcv[idx, :] = normalize(np.array([x, y, 0]))
    # 设置radius
    pos_rcv *= radius
    pos_rcv = np.concatenate([np.array([[0, 0, 0]]), pos_rcv], 0)
    return pos_rcv


def linear_array_geometry(radius: float, mic_num: int) -> np.ndarray:
    xs = np.arange(start=0, stop=radius * mic_num, step=radius)
    xs -= np.mean(xs)  # 将中心移动到原点
    pos_rcv = np.zeros((mic_num, 3))
    pos_rcv[:, 0] = xs
    return pos_rcv


def chime3_array_geometry() -> np.ndarray:
    # TODO 加入麦克风的朝向向量，以及麦克风的全向/半向
    pos_rcv = np.zeros((6, 3))
    pos_rcv[0, :] = np.array([-0.1, 0.095, 0])
    pos_rcv[1, :] = np.array([0, 0.095, 0])
    pos_rcv[2, :] = np.array([0.1, 0.095, 0])
    pos_rcv[3, :] = np.array([-0.1, -0.095, 0])
    pos_rcv[4, :] = np.array([0, -0.095, 0])
    pos_rcv[5, :] = np.array([0.1, -0.095, 0])

    # 验证边长是否正确，边与边之间是否垂直
    assert np.isclose(
        np.linalg.norm(pos_rcv[0, :] - pos_rcv[1, :]), 0.1
    ), "distance between #1 and #2 is wrong"
    assert np.isclose(
        np.linalg.norm(pos_rcv[1, :] - pos_rcv[2, :]), 0.1
    ), "distance between #2 and #3 is wrong"
    assert np.isclose(
        np.linalg.norm(pos_rcv[0, :] - pos_rcv[3, :]), 0.19
    ), "distance between #1 and #4 is wrong"
    assert np.isclose(
        np.linalg.norm(pos_rcv[2, :] - pos_rcv[5, :]), 0.19
    ), "distance between #3 and #6 is wrong"
    assert np.isclose(
        np.linalg.norm(pos_rcv[3, :] - pos_rcv[4, :]), 0.1
    ), "distance between #4 and #5 is wrong"
    assert np.isclose(
        np.linalg.norm(pos_rcv[4, :] - pos_rcv[5, :]), 0.1
    ), "distance between #5 and #6 is wrong"
    assert np.isclose(
        np.dot(pos_rcv[0, :] - pos_rcv[1, :], pos_rcv[0, :] - pos_rcv[3, :]), 0
    ), "not vertical"
    assert np.isclose(
        np.dot(pos_rcv[2, :] - pos_rcv[5, :], pos_rcv[4, :] - pos_rcv[5, :]), 0
    ), "not vertical"
    return pos_rcv


def libricss_array_geometry() -> np.ndarray:
    pos_rcv = np.zeros((7, 3))
    pos_rcv_c = circular_array_geometry(radius=0.0425, mic_num=6)
    pos_rcv[1:, :] = pos_rcv_c
    return pos_rcv


def rotate(
    pos_rcv: np.ndarray,
    x_angle: Optional[float] = None,
    y_angle: Optional[float] = None,
    z_angle: Optional[float] = None,
) -> np.ndarray:
    # 将以原点为中心的麦克风分别绕X、Y、Z轴旋转给定角度(单位：rad)
    def _rotate(pos_rcv: np.ndarray, angle: float, dims: Tuple[int, int]) -> np.ndarray:
        assert len(set(dims)) == 2, "dims参数应该给两个不同的值"
        pos_rcv_new = np.empty_like(pos_rcv)
        pos_rcv_new[:, dims[0]] = pos_rcv[:, dims[0]] * np.cos(angle) - pos_rcv[
            :, dims[1]
        ] * np.sin(angle)
        pos_rcv_new[:, dims[1]] = pos_rcv[:, dims[0]] * np.sin(angle) + pos_rcv[
            :, dims[1]
        ] * np.cos(angle)
        dim2 = list({0, 1, 2} - set(dims))[0]
        pos_rcv_new[:, dim2] = pos_rcv[:, dim2]  # 旋转轴的值在旋转前后应该是相等的

        # check angle
        norm2d = norm(pos_rcv[:, dims], axis=-1)
        real_angles = []
        for i, n in enumerate(norm2d):
            if n == 0:
                real_angles.append(None)
            else:
                real_angles.append(
                    np.arccos(
                        np.clip(
                            (pos_rcv[i, dims] * pos_rcv_new[i, dims]).sum(axis=-1)
                            / (n**2),
                            a_min=-1,
                            a_max=1,
                        )
                    )
                )
        # move angle to range [-pi, pi]
        angle = angle % (2 * np.pi)
        for i, ra in enumerate(real_angles):
            if np.isclose(norm2d[i], 0):
                continue  # skip angle check if the point is close to orgin
            assert np.isclose(ra, angle) or np.isclose(ra + angle, np.pi * 2), (
                ra,
                angle,
            )
        # check relative distance
        dist_old = norm(
            pos_rcv[:, np.newaxis, :] - pos_rcv[np.newaxis, :, :], axis=-1
        )  # shape [M, M]
        dist_new = norm(
            pos_rcv_new[:, np.newaxis, :] - pos_rcv_new[np.newaxis, :, :], axis=-1
        )  # shape [M, M]
        assert np.allclose(dist_old, dist_new)
        return pos_rcv_new

    for ang, dims in zip([x_angle, y_angle, z_angle], [(1, 2), (2, 0), (0, 1)]):
        if ang is not None:
            pos_rcv = _rotate(pos_rcv=pos_rcv, angle=ang, dims=dims)
    return pos_rcv


def generate_rir_cfg_list(
    index: Optional[int] = None,
    spk_num: int = 1,
    noise_num: int = 0,
    room_size_lims: Tuple[
        Tuple[float, float], Tuple[float, float], Tuple[float, float]
    ] = ((3, 8), (3, 8), (3, 5)),
    room_size_list: Optional[List[List[float]]] = None,
    mic_zlim: Tuple[float, float] = (1.0, 2),
    spk_zlim: Tuple[float, float] = (1.0, 2),
    RT60_lim: Tuple[float, float] = (0.1, 1.0),
    rir_nums: Tuple[int, int, int] = (40000, 5000, 3000),
    arr_geometry: Union[
        Literal["circular"],
        Literal["circular_with_center"],
        Literal["linear"],
        Literal["chime3"],
        Literal["libricss"],
    ] = "linear",
    arr_radius: Tuple[float, float] = (0.1, 0.1),
    arr_rotate_lims: Union[
        Tuple[
            Optional[Tuple[float, float]],
            Optional[Tuple[float, float]],
            Optional[Tuple[float, float]],
        ],
        Literal["auto"],
    ] = "auto",
    arr_room_center_dist: Union[float, Literal["auto"]] = "auto",
    wall_abs_weights_lims: Union[
        List[Tuple[float, float]], Literal["auto"], Literal[None]
    ] = "auto",
    mic_num: int = 1,
    mic_pos_var: float = 0,
    spk_arr_dist: Union[
        Tuple[float, float], Tuple[float, None], Literal["auto"], Literal["random"]
    ] = "auto",
    fs: int = 16000,
    attn_diff: Tuple[Optional[float], Optional[float]] = (None, None),
    save_to: Union[Literal["auto"], str] = "auto",
    rir_dir: str = "dataset/rirs_generated",
    seed: int = 2024,
    min_dist_to_wall: int = 1,
    mic_center: Optional[List[float]] = None,
    mic_center_ratio: Optional[List[float]] = None,
    mic_center_list: Optional[List[List[float]]] = None,
    restrict_src_to_inward_halfspace: bool = False,
    max_src_tries: int = 1000,
    uniform_angle_radius: bool = False,
):
    """configuration file generation

    Args:
        index: the index of one sample, please give None for this parameter
        spk_num: the number of speakers.
        noise_num: the number of point noises.
        room_size_lims: the x, y, z range of room.
        room_size_list: fixed room sizes to cycle through evenly (each item must have a mic_center_list entry).
        mic_zlim: the z range of microphone center.
        spk_zlim: the height range of speaker.
        RT60_lim: the range of RT60.
        rir_nums: the number of training/validation/test set rirs.
        arr_geometry: 'circular', 'linear' or 'chime3'.
        arr_radius: the range of radius of array.
        arr_rotate_lims: rotation angle range for x/y/z axis.
        arr_room_center_dist: the max distance between the center of array and room
        wall_abs_weights_lims: the weights of wall absorbtion coefficients. TODO: add half-open,
        mic_num: the number of microphones.
        mic_pos_var: microphone array position variation (m).
        spk_arr_dist: the distance range between the center of microphone array and speaker.
        fs: sample rate.
        attn_diff: starts from what attenuation (dB) to use diffuse model to generate rir for speech and noise. diffuse model will speed up the simulation but is not accurate?
        save_to: save the configuration file to.
        rir_dir: the dir to save generated rirs
        seed: the random seeds.
        mic_center_list: fixed mic centers aligned with room_size_list.
        mic_center_ratio: fixed mic center as ratios of room dimensions [rx, ry, rz], each in [0,1].
        uniform_angle_radius: if True, sample source positions in polar coordinates
            around mic_center so that the folded angle (in [0, 180] deg, matching
            compute_angle_radius with room_sz) and the radius (in meters) are each
            uniformly distributed over their valid ranges. Requires spk_arr_dist
            to be an explicit (min, max) tuple. Rationale: plain Cartesian (x, y)
            sampling gives a radial density that grows as ~r (annular area effect),
            so small and large radii are both undersampled; combined with
            asymmetric mic_center placement and wall-clearance rejection, it also
            produces uneven angle marginals. Uniform (angle, radius) sampling is
            what you want for a localization training set that doesn't need
            class-reweighted losses to compensate for data imbalance.
    """
    def _as_vec3(x, name: str) -> Optional[np.ndarray]:
        if x is None:
            return None
        arr = np.array(x, dtype=np.float64).reshape(-1)
        if arr.size != 3:
            raise ValueError(f"{name} must have 3 elements, got {arr.size}: {x}")
        return arr

    def _as_vec3_list(xs, name: str) -> Optional[List[np.ndarray]]:
        if xs is None:
            return None
        if not isinstance(xs, (list, tuple)) or len(xs) == 0:
            raise ValueError(f"{name} must be a non-empty list of 3-element lists.")
        out = []
        for i, x in enumerate(xs):
            out.append(_as_vec3(x, f"{name}[{i}]"))
        return out

    def _as_ratio3(x, name: str) -> Optional[np.ndarray]:
        arr = _as_vec3(x, name)
        if arr is None:
            return None
        if np.any(arr < 0.0) or np.any(arr > 1.0):
            raise ValueError(f"{name} must be ratios in [0,1], got {arr.tolist()}")
        return arr

    def _required_mic_center_margin(array_extent_xy: float, mic_pos_var: float) -> float:
        # We allow the array to be close to a wall, but it must remain inside the room.
        # Use array_extent_xy (max XY distance from center to any mic) plus mic_pos_var.
        return float(array_extent_xy + mic_pos_var)

    def _is_mic_center_valid_for_room(mic_center: np.ndarray, room_sz: List[float], margin: float) -> bool:
        # Enforce x/y wall clearance; mic_zlim handles z sampling constraints.
        if mic_center is None:
            return True
        x, y, z = mic_center.tolist()
        Lx, Ly, Lz = room_sz
        if not (0.0 <= x <= Lx and 0.0 <= y <= Ly and 0.0 <= z <= Lz):
            return False
        if (x < margin) or (y < margin) or (x > Lx - margin) or (y > Ly - margin):
            return False
        return True

    def _nearest_wall_halfspace(mic_center: np.ndarray, room_sz: List[float]):
        # Returns a callable predicate f(pos)->bool implementing inward half-space constraint
        x, y, _ = mic_center.tolist()
        Lx, Ly, _ = room_sz
        dists = {
            "x0": x,
            "xL": Lx - x,
            "y0": y,
            "yL": Ly - y,
        }
        wall = min(dists, key=dists.get)
        if wall == "x0":
            return wall, (lambda p: p[0] >= mic_center[0])
        if wall == "xL":
            return wall, (lambda p: p[0] <= mic_center[0])
        if wall == "y0":
            return wall, (lambda p: p[1] >= mic_center[1])
        assert wall == "yL"
        return wall, (lambda p: p[1] <= mic_center[1])

    if index is None:
        # set parameters and start multiprocessing generation
        assert arr_geometry in [
            "circular",
            "circular_with_center",
            "linear",
            "chime3",
            "libricss",
        ], "only supports circular, circular_with_center, linear, chime3 and libricss array for now"

        if (
            arr_geometry == "circular"
            or arr_geometry == "linear"
            or arr_geometry == "circular_with_center"
        ):
            if arr_rotate_lims == "auto":
                arr_rotate_lims = (None, None, (0, 2 * np.pi))  # rotate by z-axis only
            if spk_arr_dist == "auto":
                spk_arr_dist = "random"
            if arr_room_center_dist == "auto":
                arr_room_center_dist = 0.5
        elif arr_geometry == "chime3":
            if arr_rotate_lims == "auto":
                arr_rotate_lims = ((0, 2 * np.pi), (0, 2 * np.pi), (0, 2 * np.pi))
            if spk_arr_dist == "auto":
                spk_arr_dist = (0.3, 0.5)
            if arr_room_center_dist == "auto":
                arr_room_center_dist = 2.0
        elif arr_geometry == "libricss":
            arr_radius = (0.0425, 0.0425)
            mic_num = 7
            if arr_rotate_lims == "auto":
                arr_rotate_lims = (None, None, (0, 2 * np.pi))  # rotate by z-axis only
            if spk_arr_dist == "auto":
                spk_arr_dist = (0.5, 4.5)
            if arr_room_center_dist == "auto":
                arr_room_center_dist = 1.0

        if wall_abs_weights_lims == "auto":
            wall_abs_weights_lims = [(0.5, 1.0)] * 6
        elif wall_abs_weights_lims is None:
            wall_abs_weights_lims = [(1.0, 1.0)] * 6
        else:
            assert (
                len(wall_abs_weights_lims) == 6
            ), "you should give the weights of six walls"
        if save_to == "auto":
            save_to = os.path.join(rir_dir, "rir_cfg.npz")
        # capture only actual configuration parameters (avoid local callables for pickling)
        args = (
            locals().copy()
        )  # start from locals, then drop helpers/callables
        for _k in list(args.keys()):
            if callable(args[_k]) or _k.startswith("_"):
                del args[_k]

        if room_size_list is not None:
            room_size_list_checked = _as_vec3_list(room_size_list, "room_size_list")
            mic_center_list_checked = _as_vec3_list(mic_center_list, "mic_center_list")
            if mic_center_list_checked is None:
                raise ValueError("mic_center_list must be provided when room_size_list is set.")
            if len(room_size_list_checked) != len(mic_center_list_checked):
                raise ValueError(
                    f"room_size_list length {len(room_size_list_checked)} must match mic_center_list length {len(mic_center_list_checked)}."
                )
        if mic_center_ratio is not None and room_size_list is not None:
            raise ValueError("mic_center_ratio cannot be used with room_size_list/mic_center_list.")
        if mic_center_ratio is not None and mic_center is not None:
            raise ValueError("mic_center_ratio cannot be used with mic_center.")

        if os.path.exists(save_to):
            cfg = dict(np.load(save_to, allow_pickle=True))
            print("load rir cfgs from file " + save_to)
            print("Args in npz: \n", cfg["args"].item())
            return cfg
        else:
            print("Args:")
            print(dict(args), "\n")

        # Pre-flight diagnostic for uniform_angle_radius mode: surface the
        # geometric upper bound on r_hi that the rooms can actually support
        # at every angle. If r_hi exceeds this bound, the radius marginal
        # will taper even with perfect (angle, radius) targeting because
        # some (theta, r) cells are physically outside every room. Printing
        # this at setup (rather than only leaving it for the final plot) so
        # the user can retune before burning compute on a mismatched config.
        if uniform_angle_radius:
            if isinstance(spk_arr_dist, (tuple, list)) and len(spk_arr_dist) == 2 and spk_arr_dist[1] is not None:
                r_lo_check, r_hi_check = float(spk_arr_dist[0]), float(spk_arr_dist[1])
                # For a ratio-placed mic with the Lx<->Ly swap guarantee, the
                # tightest radial constraint is along the x axis.  With the
                # swap, Lx >= Ly, the mic ratio along x is mic_center_ratio[0]
                # (typically < 0.5), and the binding wall is whichever of
                # (x0, xL) is further (we want *both* walls reachable from
                # the mic at radius r_hi, so the tighter side is min of the
                # two mic-to-wall distances).
                if mic_center_ratio is not None and room_size_lims is not None:
                    rx = float(mic_center_ratio[0])
                    # After swap, min_x >= min_y, so the x-axis direction is
                    # the binding one. x_ratio_tight is the smaller of the two
                    # x-wall distances expressed as a fraction of Lx.
                    x_ratio_tight = min(rx, 1.0 - rx)
                    xlim_pre = room_size_lims[0]
                    lx_min = float(xlim_pre[0])
                    lx_max = float(xlim_pre[1])
                    r_max_worst = x_ratio_tight * lx_min - min_dist_to_wall
                    r_max_best = x_ratio_tight * lx_max - min_dist_to_wall
                    # Lx threshold above which r_hi is reachable at theta=0/180;
                    # fraction of the Lx range that lies above that threshold
                    # approximates the fraction of rooms in which the target
                    # (angle, radius) cell at the extreme angles is feasible.
                    lx_thresh = (r_hi_check + min_dist_to_wall) / x_ratio_tight
                    frac_full_support = max(
                        0.0,
                        min(1.0, (lx_max - lx_thresh) / max(1e-9, (lx_max - lx_min))),
                    )
                    print(
                        "[uniform_angle_radius] geometric bound on uniform radius "
                        "(ratio-placed mic, post Lx<->Ly swap):"
                    )
                    print(
                        f"  mic_center_ratio[0]={rx:.3f} -> x-wall tighter ratio={x_ratio_tight:.3f}, "
                        f"room Lx in [{lx_min}, {lx_max}] -> r_max at theta=0/180 in "
                        f"[{r_max_worst:.2f}, {r_max_best:.2f}] m"
                    )
                    print(
                        f"  spk_arr_dist (user-requested) = [{r_lo_check}, {r_hi_check}] m"
                    )
                    print(
                        f"  ~{frac_full_support*100:.0f}% of sampled rooms fully support r_hi={r_hi_check:.2f} m "
                        f"at all angles (Lx >= {lx_thresh:.2f} m); the rest rely on phase-2 "
                        f"relaxation (angle target kept, r free) and contribute to the "
                        f"small-r tail."
                    )
                    if frac_full_support < 0.5:
                        print(
                            f"  NOTE: fewer than half the rooms can fit r_hi={r_hi_check} m at "
                            f"all angles.  This still yields an EXACTLY uniform angle "
                            f"marginal (phase-1 guarantees every angle bin is attempted), "
                            f"but the radius marginal will droop above ~{r_max_worst:.2f} m. "
                            f"If flatter radius matters more than coverage, either:"
                        )
                        print(
                            f"    (a) raise room_size_lims x-range to min Lx >= {lx_thresh:.1f} m (loses small-room variety)"
                        )
                        print(
                            f"    (b) lower spk_arr_dist[1] (e.g. to ~{(x_ratio_tight*((lx_min+lx_max)/2) - min_dist_to_wall):.2f} m, the mean-room r_max)"
                        )
                        print(
                            f"    (c) use a more centered mic_center_ratio (e.g. [0.5, ry, rz])"
                        )
                    print()

        rir_num = sum(rir_nums)
        print("generating rir cfgs. ", end=" ")
        import time

        ts = time.time()
        with multiprocessing.Pool(processes=multiprocessing.cpu_count() // 2) as p:
            new_args = args.copy()
            del new_args["index"]
            rir_pars = p.map(partial(generate_rir_cfg_list, **new_args), range(rir_num))
        print("time used: ", time.time() - ts)

        cfg = {
            "args": np.array(args),
            "rir_pars": rir_pars,
        }

        # save to npz
        dir = os.path.dirname(save_to)
        if len(dir) > 0:
            os.makedirs(dir, exist_ok=True)
        np.savez_compressed(save_to, **cfg)
        return cfg

    # generate one room cfg
    np.random.seed(seed=seed + index)
    xlim, ylim, zlim = room_size_lims

    # sample radius / RT60 / room_sz / abs_weights
    array_r_this = uniform(*arr_radius)
    mic_center = _as_vec3(mic_center, "mic_center")
    mic_center_ratio = _as_ratio3(mic_center_ratio, "mic_center_ratio")
    room_size_list = _as_vec3_list(room_size_list, "room_size_list")
    mic_center_list = _as_vec3_list(mic_center_list, "mic_center_list")

    # Determine array extent in XY (max distance from center to any mic), which is the key constraint
    # for keeping a fixed mic array inside room boundaries.
    if arr_geometry == "circular":
        pos_rcv_local = circular_array_geometry(radius=array_r_this, mic_num=mic_num)
    elif arr_geometry == "circular_with_center":
        pos_rcv_local = circular_with_center_array_geometry(radius=array_r_this, mic_num=mic_num)
    elif arr_geometry == "linear":
        pos_rcv_local = linear_array_geometry(radius=array_r_this, mic_num=mic_num)
    elif arr_geometry == "chime3":
        pos_rcv_local = chime3_array_geometry()
    else:
        assert arr_geometry == "libricss", arr_geometry
        pos_rcv_local = libricss_array_geometry()

    array_extent_xy = float(np.max(norm(pos_rcv_local[:, :2], axis=-1))) if len(pos_rcv_local) > 0 else 0.0
    req_margin = _required_mic_center_margin(array_extent_xy, mic_pos_var)

    if room_size_list is not None:
        if mic_center_list is None:
            raise ValueError("mic_center_list must be provided when room_size_list is set.")
        if len(room_size_list) != len(mic_center_list):
            raise ValueError(
                f"room_size_list length {len(room_size_list)} must match mic_center_list length {len(mic_center_list)}."
            )
        list_idx = index % len(room_size_list)
        room_sz = room_size_list[list_idx].tolist()
        mic_center = mic_center_list[list_idx]
        if not _is_mic_center_valid_for_room(mic_center, room_sz, req_margin):
            raise ValueError(
                f"mic_center_list[{list_idx}]={mic_center.tolist()} does not fit room_size_list[{list_idx}]={room_sz} "
                f"with margin={req_margin:.3f}."
            )
        RT60 = uniform(*RT60_lim)
        while is_valid_RT60_for_room(room_sz, RT60) == False:
            RT60 = uniform(*RT60_lim)
    else:
        RT60 = uniform(*RT60_lim)  # sample a RT60
        room_sz = [uniform(*xlim), uniform(*ylim), uniform(*zlim)]  # sample a room
        mic_center_ratio_tmp = (
            mic_center_ratio * np.array(room_sz, dtype=np.float64)
            if mic_center_ratio is not None
            else None
        )
        # resample if the RT60 could not be satisfied in this room OR fixed mic center cannot fit this array
        while (is_valid_RT60_for_room(room_sz, RT60) == False) or (
            mic_center is not None and (not _is_mic_center_valid_for_room(mic_center, room_sz, req_margin))
        ) or (
            mic_center_ratio_tmp is not None
            and (not _is_mic_center_valid_for_room(mic_center_ratio_tmp, room_sz, req_margin))
        ):
            room_sz = [uniform(*xlim), uniform(*ylim), uniform(*zlim)]
            RT60 = uniform(*RT60_lim)
            mic_center_ratio_tmp = (
                mic_center_ratio * np.array(room_sz, dtype=np.float64)
                if mic_center_ratio is not None
                else None
            )
        if mic_center_ratio_tmp is not None:
            mic_center = mic_center_ratio_tmp
        # When uniform_angle_radius is enabled with a ratio-placed mic, ensure
        # the y-wall is the one closest to the mic. Otherwise the inward
        # halfspace predicate restricts the accessible folded angle to a
        # *half* of [0, 180] (x-wall nearest -> only [0, 90] accessible;
        # xL-wall nearest -> only [90, 180] accessible), which produces the
        # exact ~1.8:1 asymmetry seen in early runs. Swapping Lx<->Ly gives
        # an equivalent rotated room in which the y-wall is now closer; this
        # preserves the physical set of rooms when xlim == ylim (the common
        # case), which you should validate yourself if your xlim != ylim.
        if uniform_angle_radius and mic_center_ratio is not None:
            mc = mic_center
            Lx, Ly = room_sz[0], room_sz[1]
            min_y = min(mc[1], Ly - mc[1])
            min_x = min(mc[0], Lx - mc[0])
            if min_y > min_x:
                room_sz = [room_sz[1], room_sz[0], room_sz[2]]
                mic_center = mic_center_ratio * np.array(room_sz, dtype=np.float64)
    # sample abs_weights then compute reflection coefficients
    abs_weights = [uniform(*abs_lim) for abs_lim in wall_abs_weights_lims]
    beta, t60error = beta_SabineEstimation(room_sz, RT60, abs_weights=abs_weights)
    while t60error > 0.05:
        abs_weights = [uniform(*abs_lim) for abs_lim in wall_abs_weights_lims]
        beta, t60error = beta_SabineEstimation(room_sz, RT60, abs_weights=abs_weights)
    # if not is_valid_beta(beta):  #  t60error > 0.05:
    #     warnings.warn(f'the given RT60={RT60} could not achieved with the given room_sz={room_sz} and abs_weights={abs_weights}')

    # microphone positions
    if mic_center is None:
        mic_center_tmp = None
        while (
            mic_center_tmp is None
            or mic_center_tmp[0] < 0.5
            or mic_center_tmp[1] < 0.5
            or mic_center_tmp[0] > room_sz[0] - 0.5
            or mic_center_tmp[1] > room_sz[1] - 0.5
        ):
            mic_center_tmp = np.array(
                [
                    uniform(
                        max(min_dist_to_wall, room_sz[0] / 2 - arr_room_center_dist),
                        min(
                            room_sz[0] - min_dist_to_wall,
                            room_sz[0] / 2 + arr_room_center_dist,
                        ),
                    ),
                    uniform(
                        max(min_dist_to_wall, room_sz[1] / 2 - arr_room_center_dist),
                        min(
                            room_sz[1] - min_dist_to_wall,
                            room_sz[1] / 2 + arr_room_center_dist,
                        ),
                    ),
                    uniform(*mic_zlim),
                ]
            )
        mic_center = mic_center_tmp
    else:
        # Fixed mic center: still ensure z is within room bounds. We already enforced x/y margin above.
        if not (0.0 <= mic_center[2] <= room_sz[2]):
            raise ValueError(
                f"mic_center z={mic_center[2]} is outside room z-range [0,{room_sz[2]}]."
            )
    pos_rcv = pos_rcv_local.copy()

    # rotate the array by x/y/z axis
    x_angle, y_angle, z_angle = [
        None if lim is None else uniform(*lim) for lim in arr_rotate_lims
    ]
    pos_rcv = rotate(pos_rcv=pos_rcv, x_angle=x_angle, y_angle=y_angle, z_angle=z_angle)

    # move center from origin to mic_center
    pos_rcv += mic_center[np.newaxis, :]

    # add small position variations to the (x,y,z) of each mic for simulating position's inperfection
    if mic_pos_var > 0:
        pos_rcv = pos_rcv + uniform(
            low=-mic_pos_var, high=mic_pos_var, size=pos_rcv.shape
        )

    # sample speaker postions
    pos_src = np.empty((spk_num, 3))
    # all speaker's loc are randomly sampled
    wall_pred = None
    if restrict_src_to_inward_halfspace and mic_center is not None:
        _, wall_pred = _nearest_wall_halfspace(mic_center=mic_center, room_sz=room_sz)

    if uniform_angle_radius:
        # Theta round-robin targeting with two radius modes:
        #
        # Fixed-r mode  (spk_arr_dist = [r_lo, r_hi], both floats):
        #   - theta: round-robin across n_th_bins=36 equal bins → flat angle marginal.
        #   - r:     round-robin across n_r_bins=37 bins on [r_lo, r_hi] (coprime with
        #     36 so joint (theta_bin, r_bin) de-correlates).  When the full target
        #     (theta, r) cell is infeasible the r target is relaxed (phase 2),
        #     keeping the angle marginal flat.
        #
        # Full-room mode  (spk_arr_dist = [r_lo, None]):
        #   - theta: same round-robin → flat angle marginal guaranteed.
        #   - r:     per-physical-angle r_max computed from the room boundary;
        #     r sampled uniformly in [r_lo, r_max(theta_phys)].  No r round-robin
        #     is needed because every sample automatically uses the full available
        #     range at that angle.  Phase 2 (r relaxation) is also unnecessary.
        #
        # The angle wall-asymmetry is handled by the Lx<->Ly swap above.
        if mic_center is None:
            raise ValueError("uniform_angle_radius=True requires a known mic_center.")
        if not isinstance(spk_arr_dist, (tuple, list)) or len(spk_arr_dist) != 2:
            raise ValueError(
                "uniform_angle_radius=True requires spk_arr_dist=(r_min, r_max) "
                "where r_max may be None for full-room mode; "
                f"got {spk_arr_dist!r}."
            )
        r_lo = float(spk_arr_dist[0])
        full_room_mode = spk_arr_dist[1] is None
        r_hi = None if full_room_mode else float(spk_arr_dist[1])
        if full_room_mode:
            if not (r_lo >= 0.0):
                raise ValueError(
                    f"uniform_angle_radius full-room mode requires r_min >= 0; got {r_lo}."
                )
        else:
            if not (r_lo >= 0.0 and r_hi > r_lo):
                raise ValueError(
                    f"uniform_angle_radius requires 0 <= r_min < r_max; got {(r_lo, r_hi)}."
                )

        n_th_bins = 36
        th_bin = index % n_th_bins
        th_tgt_lo = th_bin * (180.0 / n_th_bins)
        th_tgt_hi = (th_bin + 1) * (180.0 / n_th_bins)

        # Radius bin round-robin only applies in fixed-r mode.
        if not full_room_mode:
            n_r_bins = 37  # coprime with 36 so joint (theta_bin, r_bin) de-correlates
            r_bin = index % n_r_bins
            r_tgt_lo = r_lo + r_bin * (r_hi - r_lo) / n_r_bins
            r_tgt_hi = r_lo + (r_bin + 1) * (r_hi - r_lo) / n_r_bins

        def _try_place(theta_range, r_range):
            """One rejection-sampling attempt.

            theta_range: (lo_deg, hi_deg) for the folded angle.
            r_range:     (r_lo, r_hi) for fixed-r mode, or None for full-room
                         mode in which r_max is computed per physical-angle
                         direction from the room boundary.

            Returns the sampled 3-D position if feasible, else None.
            """
            theta_folded_deg = uniform(*theta_range)
            z = uniform(spk_zlim[0], spk_zlim[1])
            if not (0.0 <= z <= room_sz[2]):
                return None
            for side in np.random.permutation(2):
                theta_phys_deg = (
                    theta_folded_deg if side == 0 else 360.0 - theta_folded_deg
                )
                theta_phys_rad = np.deg2rad(theta_phys_deg)

                if r_range is None:
                    # Full-room mode: derive r_hi from room geometry at this angle.
                    r_hi_eff = _rmax_at_angle(
                        float(mic_center[0]), float(mic_center[1]),
                        room_sz[0], room_sz[1], theta_phys_rad, min_dist_to_wall,
                    )
                    if r_hi_eff <= r_lo:
                        continue
                    r = uniform(r_lo, r_hi_eff)
                else:
                    r = uniform(*r_range)

                cand = np.array(
                    [
                        float(mic_center[0]) + r * np.cos(theta_phys_rad),
                        float(mic_center[1]) + r * np.sin(theta_phys_rad),
                        z,
                    ]
                )
                # Safety wall-clearance guard (always kept even in full-room mode).
                if cand[0] < min_dist_to_wall or cand[0] > room_sz[0] - min_dist_to_wall:
                    continue
                if cand[1] < min_dist_to_wall or cand[1] > room_sz[1] - min_dist_to_wall:
                    continue
                if wall_pred is not None and not wall_pred(cand):
                    continue
                return cand
            return None

        for iiii in range(0, spk_num):
            # Only the first source participates in the round-robin targeting;
            # extra sources for spk_num > 1 are sampled unconstrained.
            if iiii != 0:
                cand = None
                r_arg_free = None if full_room_mode else (r_lo, r_hi)
                for _ in range(max_src_tries):
                    cand = _try_place((0.0, 180.0), r_arg_free)
                    if cand is not None:
                        break
                if cand is None:
                    raise RuntimeError(
                        f"uniform_angle_radius: failed to place secondary source "
                        f"after {max_src_tries} tries (index={index})."
                    )
                pos_src[iiii, :] = cand
                continue

            cand = None
            if full_room_mode:
                # Phase 1: target theta-bin; r_max derived from room boundary.
                # No phase 2 r-relaxation needed — r is already maximally free.
                for _ in range(max_src_tries // 2):
                    cand = _try_place((th_tgt_lo, th_tgt_hi), None)
                    if cand is not None:
                        break
            else:
                # Phase 1: strict target theta-bin AND target r-bin.
                for _ in range(max_src_tries // 4):
                    cand = _try_place((th_tgt_lo, th_tgt_hi), (r_tgt_lo, r_tgt_hi))
                    if cand is not None:
                        break
                if cand is None:
                    # Phase 2: keep theta-bin target, relax r to full range.
                    # This preserves the angle marginal exactly while letting the
                    # room dictate what radii are achievable at this angle.
                    for _ in range(max_src_tries // 2):
                        cand = _try_place((th_tgt_lo, th_tgt_hi), (r_lo, r_hi))
                        if cand is not None:
                            break

            if cand is None:
                # Phase 3: ultimate fallback, unconstrained uniform rejection.
                # Reached only when even the target theta-bin is geometrically
                # infeasible in this room, which shouldn't happen with the
                # wall-swap guarantee unless the user deliberately fixed
                # `mic_center` instead of using ratio.
                r_arg_free = None if full_room_mode else (r_lo, r_hi)
                for _ in range(max_src_tries):
                    cand = _try_place((0.0, 180.0), r_arg_free)
                    if cand is not None:
                        break
            if cand is None:
                raise RuntimeError(
                    f"uniform_angle_radius: all phases failed after ~{max_src_tries} tries "
                    f"(index={index}, room_sz={room_sz}, mic_center={mic_center.tolist()}, "
                    f"spk_arr_dist=({r_lo}, {r_hi!r})). "
                    f"Enlarge rooms, adjust mic_center_ratio, or raise max_src_tries."
                )
            pos_src[iiii, :] = cand
    else:
        for iiii in range(0, spk_num):
            tries = 0
            while True:
                pos_src[iiii, :] = (
                    uniform(min_dist_to_wall, room_sz[0] - min_dist_to_wall),
                    uniform(min_dist_to_wall, room_sz[1] - min_dist_to_wall),
                    uniform(spk_zlim[0], spk_zlim[1]),
                )
                if wall_pred is not None and (not wall_pred(pos_src[iiii, :])):
                    tries += 1
                    if tries >= max_src_tries:
                        raise RuntimeError(
                            f"Failed to sample a source position satisfying inward-halfspace after {max_src_tries} tries. "
                            f"Try increasing room_size_lims, adjusting mic_center, or disabling restrict_src_to_inward_halfspace."
                        )
                    continue
                break
            if spk_arr_dist == "random":
                continue
            while (
                norm(pos_src[iiii, :] - mic_center) < spk_arr_dist[0]
                or norm(pos_src[iiii, :] - mic_center) > spk_arr_dist[1]
                or (wall_pred is not None and (not wall_pred(pos_src[iiii, :])))
            ):
                # if the spk_mic_dis requirements are not satisfied, then resample a position
                pos_src[iiii, :] = (
                    uniform(min_dist_to_wall, room_sz[0] - min_dist_to_wall),
                    uniform(min_dist_to_wall, room_sz[1] - min_dist_to_wall),
                    uniform(spk_zlim[0], spk_zlim[1]),
                )
                tries += 1
                if tries >= max_src_tries:
                    raise RuntimeError(
                        f"Failed to sample a source position satisfying constraints after {max_src_tries} tries. "
                        f"Try loosening spk_arr_dist, increasing room_size_lims, adjusting mic_center, or disabling restrict_src_to_inward_halfspace."
                    )

    # generate the positions of noise sources
    pos_noise = []
    for iiii in range(noise_num):
        pos_noise.append([uniform(0.1, sz - 0.1) for sz in room_sz])
    pos_noise = np.array(pos_noise)

    # plot room
    # print(x_angle / np.pi * 180, y_angle / np.pi * 180, z_angle / np.pi * 180)
    # plot_room(room_sz=room_sz, pos_src=pos_src, pos_rcv=pos_rcv, pos_noise=pos_noise, saveto=None)

    print(room_sz,pos_src,pos_rcv)
    par = {
        "index": index,
        "RT60": RT60,
        "room_sz": room_sz,
        "pos_src": pos_src.astype(np.float32),
        "pos_rcv": pos_rcv.astype(np.float32),
        "pos_noise": pos_noise.astype(np.float32),
        # 'abs_weights': abs_weights,
        "beta": beta,
    }
    return par


def generate_rir_files(
    rir_cfg: Dict[str, Any],
    rir_dir: str,
    rir_nums: Tuple[int, int, int],
    use_gpu: bool,
    write_localization_csv: bool = True,
    localization_csv_path: Optional[str] = None,
):
    train_rir_num, val_rir_num, test_rir_num = rir_nums

    pars = rir_cfg["rir_pars"]
    fs = rir_cfg["args"].item()["fs"]
    attn_diff_speech = rir_cfg["args"].item()["attn_diff"]
    attn_diff_noise = None
    # print(attn_diff_speech)
    # exit()
    if isinstance(attn_diff_speech, (tuple, list)):
        attn_diff_noise = attn_diff_speech[1]
        attn_diff_speech = attn_diff_speech[0]

    if (
        (Path(rir_dir) / "train").exists()
        or (Path(rir_dir) / "validation").exists()
        or (Path(rir_dir) / "test").exists()
    ):
        ans = input("dir " + rir_dir + " exists, still generate rir?(y/n)")
        assert ans in ["y", "n"], ans
        if ans == "n":
            return
    else:
        os.makedirs(rir_dir, exist_ok=True)

    for setdir in ["train", "validation", "test"]:
        os.makedirs(os.path.join(rir_dir, setdir), exist_ok=True)

    def __gen__(par, fs, use_gpu: bool):
        index = par["index"]
        RT60 = par["RT60"]
        room_sz = par["room_sz"]
        pos_src = np.array(par["pos_src"])
        pos_rcv = np.array(par["pos_rcv"])
        pos_noise = np.array(par["pos_noise"])
        # abs_weights = np.array(par['abs_weights']) if 'abs_weights' in par else None
        beta = np.array(par["beta"]) if "beta" in par else None

        if index < train_rir_num:
            setdir = "train"
        elif index >= train_rir_num and index < train_rir_num + val_rir_num:
            setdir = "validation"
        else:
            setdir = "test"
        save_to = os.path.join(rir_dir, setdir, str(index) + ".npz")
        if os.path.exists(save_to):
            try:  # try to load, if no error is reported then skip this
                np.load(save_to, allow_pickle=True)
                return
            except:
                ...
        if not use_gpu:
            rir = generate_rir_cpu(
                room_sz, pos_src, pos_rcv, RT60, fs, beta=beta
            )  # reverbrant rir
            rir_dp = generate_rir_cpu(
                room_sz, pos_src, pos_rcv, 0, fs
            )  # direct path rir
            rir_noise = generate_rir_cpu(
                room_sz, pos_noise, pos_rcv, RT60, fs, beta=beta
            )  # noise rir
        else:
            rir = generate_rir_gpu(
                room_sz,
                pos_src,
                pos_rcv,
                RT60,
                fs,
                att_diff=attn_diff_speech,
                beta=beta,
            )  # reverbrant rir
            rir_dp = generate_rir_gpu(
                room_sz, pos_src, pos_rcv, 0, fs
            )  # direct path rir
            rir_noise = generate_rir_gpu(
                room_sz,
                pos_noise,
                pos_rcv,
                RT60,
                fs,
                att_diff=attn_diff_noise,
                beta=beta,
            )  # noise rir, uses diffuse model after 20 dB attenuation
        if rir_noise is not None:
            rir_noise = rir_noise.astype(np.float16)
        np.savez_compressed(
            save_to,
            fs=fs,
            RT60=RT60,
            room_sz=room_sz,
            pos_src=pos_src,
            pos_rcv=pos_rcv,
            pos_noise=pos_noise,
            rir=rir,
            rir_dp=rir_dp,
            rir_noise=rir_noise,
        )
        
        if rir_dp.shape[0]==1:
        
            # rir_dp_savepath = os.path.join(rir_dir,setdir,'direct_path',str(index)+'.wav')
            rir_rev_savepath = os.path.join(rir_dir,setdir,'reverb',str(index)+'.wav')
            
            # os.makedirs(os.path.join(rir_dir,setdir,'direct_path'),exist_ok=True)
            os.makedirs(os.path.join(rir_dir,setdir,'reverb'),exist_ok=True)
            
            # sf.write(rir_dp_savepath,rir_dp.squeeze(),fs)
            sf.write(rir_rev_savepath,rir.squeeze(),fs)
        

    if not use_gpu:
        from p_tqdm import p_map

        p_map(
            partial(__gen__, fs=fs, use_gpu=use_gpu),
            pars,
            num_cpus=multiprocessing.cpu_count() // 2,
        )
    else:
        pbar = tqdm.tqdm(total=len(pars))
        pbar.set_description("generating rirs")
        for i, par in enumerate(pars):
            pbar.update()
            __gen__(par=par, fs=fs, use_gpu=use_gpu)

    if write_localization_csv:
        if not localization_csv_path:
            localization_csv_path = os.path.join(rir_dir, "rir_localization.csv")
        rows = []
        for par in pars:
            index = par["index"]
            if index < train_rir_num:
                setdir = "train"
            elif index >= train_rir_num and index < train_rir_num + val_rir_num:
                setdir = "validation"
            else:
                setdir = "test"
            rir_abs_path = os.path.join(rir_dir, setdir, f"{index}.npz")
            angle_deg, radius_m = compute_angle_radius(par["pos_src"], par["pos_rcv"], par["room_sz"])
            rows.append((rir_abs_path, angle_deg, radius_m))
        with open(localization_csv_path, "w") as handle:
            handle.write("rir_path,angle_deg,radius_m\n")
            for rir_path, angle_deg, radius_m in rows:
                handle.write(f"{rir_path},{angle_deg:.6f},{radius_m:.6f}\n")


def plot_distribution(
    rir_pars,
    saveto: str,
    num_angle_bins: int = 36,
    num_radius_bins: int = 48,
    rir_nums: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Visualize the (angle, radius) distribution of a generated RIR set.

    Produces a three-panel figure:
      1. Angle histogram with a "uniform target" reference line.
      2. Radius histogram with a "uniform target" reference line.
      3. Joint (angle x radius) 2D heatmap.

    Any departure from uniformity is immediately visible as a bar deviating
    from the dashed red line, or as bright/dark stripes in the heatmap. This
    is meant as a sanity check for the `uniform_angle_radius` flag.

    rir_pars: list of per-sample dicts (as produced by generate_rir_cfg_list).
    saveto: output .png path.
    rir_nums: optional (train, val, test) split counts. If given, renders an
      extra row of histograms per split so you can confirm the split-level
      balance as well.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    angles: List[float] = []
    radii: List[float] = []
    for par in rir_pars:
        a, r = compute_angle_radius(par["pos_src"], par["pos_rcv"], par["room_sz"])
        angles.append(a)
        radii.append(r)
    angles_np = np.asarray(angles)
    radii_np = np.asarray(radii)

    r_min = 0.0
    r_max = float(np.ceil(radii_np.max() + 0.5))

    nrows = 1 if rir_nums is None else 2
    fig, axes = plt.subplots(nrows, 3, figsize=(18, 5 * nrows), squeeze=False)

    def _panel_hist(ax, values, bins, value_range, xlabel, title):
        n, _, _ = ax.hist(
            values,
            bins=bins,
            range=value_range,
            color="steelblue",
            edgecolor="black",
            linewidth=0.3,
        )
        target = len(values) / bins if bins > 0 else 0.0
        ax.axhline(
            target,
            color="red",
            ls="--",
            lw=1.2,
            label=f"uniform target ({target:.0f})",
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.legend(loc="upper right")

    _panel_hist(
        axes[0, 0],
        angles_np,
        num_angle_bins,
        (0.0, 180.0),
        "angle (deg, folded to [0,180])",
        f"Angle distribution  (N={len(angles_np)})",
    )
    _panel_hist(
        axes[0, 1],
        radii_np,
        num_radius_bins,
        (r_min, r_max),
        "radius (m)",
        f"Radius distribution  (N={len(radii_np)})",
    )
    h2 = axes[0, 2].hist2d(
        angles_np,
        radii_np,
        bins=[num_angle_bins, num_radius_bins],
        range=[[0.0, 180.0], [r_min, r_max]],
        cmap="viridis",
    )
    plt.colorbar(h2[3], ax=axes[0, 2], label="count")
    axes[0, 2].set_title("Joint (angle x radius)")
    axes[0, 2].set_xlabel("angle (deg)")
    axes[0, 2].set_ylabel("radius (m)")

    if rir_nums is not None:
        n_tr, n_va, n_te = rir_nums
        splits = [
            ("train", slice(0, n_tr)),
            ("validation", slice(n_tr, n_tr + n_va)),
            ("test", slice(n_tr + n_va, n_tr + n_va + n_te)),
        ]
        colors = {"train": "steelblue", "validation": "darkorange", "test": "seagreen"}
        # Overlay per-split histograms as step traces in the same axes row.
        for name, sl in splits:
            a_s = angles_np[sl]
            r_s = radii_np[sl]
            if len(a_s) == 0:
                continue
            axes[1, 0].hist(
                a_s,
                bins=num_angle_bins,
                range=(0.0, 180.0),
                histtype="step",
                linewidth=1.5,
                color=colors[name],
                label=f"{name} (N={len(a_s)})",
            )
            axes[1, 1].hist(
                r_s,
                bins=num_radius_bins,
                range=(r_min, r_max),
                histtype="step",
                linewidth=1.5,
                color=colors[name],
                label=f"{name} (N={len(r_s)})",
            )
        axes[1, 0].set_title("Angle distribution per split")
        axes[1, 0].set_xlabel("angle (deg)")
        axes[1, 0].set_ylabel("count")
        axes[1, 0].legend(loc="upper right")
        axes[1, 1].set_title("Radius distribution per split")
        axes[1, 1].set_xlabel("radius (m)")
        axes[1, 1].set_ylabel("count")
        axes[1, 1].legend(loc="upper right")
        axes[1, 2].axis("off")
        axes[1, 2].text(
            0.05,
            0.95,
            f"total samples: {len(angles_np)}\n"
            f"angle range observed: [{angles_np.min():.2f}, {angles_np.max():.2f}] deg\n"
            f"radius range observed: [{radii_np.min():.3f}, {radii_np.max():.3f}] m\n"
            f"angle CV across bins: {_cv_across_bins(angles_np, num_angle_bins, (0.0, 180.0)):.3f}\n"
            f"radius CV across bins: {_cv_across_bins(radii_np, num_radius_bins, (r_min, r_max)):.3f}\n"
            f"(CV = std/mean of bin counts; 0.0 = perfectly uniform)",
            transform=axes[1, 2].transAxes,
            va="top",
            ha="left",
            family="monospace",
            fontsize=10,
        )

    fig.tight_layout()
    save_dir = os.path.dirname(saveto)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    fig.savefig(saveto, dpi=120)
    plt.close(fig)
    print(f"[plot_distribution] wrote {saveto}  (N={len(angles_np)})")


def _cv_across_bins(values: np.ndarray, num_bins: int, value_range: Tuple[float, float]) -> float:
    """Coefficient of variation (std/mean) of per-bin counts in a histogram.
    0.0 => perfectly uniform. Used as a scalar uniformity score.
    """
    counts, _ = np.histogram(values, bins=num_bins, range=value_range)
    mean = counts.mean()
    if mean <= 0:
        return float("nan")
    return float(counts.std() / mean)


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=0 python generate_rirs.py --help
    parser = ArgumentParser(
        description="code for generating RIRs by Changsheng Quan @ Westlake University"
    )
    parser.add_function_arguments(
        generate_rir_cfg_list
    )  # add_argument for the function generate_rir_cfg_list
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu or not")
    parser.add_argument(
        "--write_localization_csv",
        type=bool,
        default=True,
        help="write CSV with rir_path/angle/radius metadata",
    )
    parser.add_argument(
        "--localization_csv_path",
        type=str,
        default=None,
        help="output CSV path (default: <rir_dir>/rir_localization.csv)",
    )
    parser.add_argument(
        "-c", "--config", required=False, type=str, help="Configuration .json file"
    )
    # NOTE: mic_center / restrict_src_to_inward_halfspace / max_src_tries are exposed via add_function_arguments(generate_rir_cfg_list)
    args = parser.parse_args()
    if args.config:
        with open(args.config, "r") as json_cfg:
            json_arg = json.load(json_cfg)
            vars(args).update(json_arg)


    # get paramters for function `generate_rir_cfg_list`
    sig = inspect.signature(generate_rir_cfg_list)
    args_for_generate_rir_cfg_list = dict()
    for param in sig.parameters.values():
        args_for_generate_rir_cfg_list[param.name] = getattr(args, param.name)


    os.makedirs(args.rir_dir, exist_ok=True)
    save_config_to_file(args, os.path.join(args.rir_dir, "config.json"))

    

    # generate configuration
    rir_cfg = generate_rir_cfg_list(**args_for_generate_rir_cfg_list)




    # generate rir files
    generate_rir_files(
        rir_cfg=rir_cfg,
        rir_dir=args.rir_dir,
        rir_nums=args.rir_nums,
        use_gpu=args.use_gpu,
        write_localization_csv=args.write_localization_csv,
        localization_csv_path=args.localization_csv_path,
    )

    # Sanity-check plot: verifies that the (angle, radius) marginals are flat
    # after generation. Lives at <rir_dir>/data_distribution.png so it sits
    # next to the generated RIRs for easy review. Works whether or not
    # uniform_angle_radius was on (lets you inspect the imbalance on legacy
    # runs too).
    pars_list = rir_cfg["rir_pars"]
    # np.load turns lists into 0-d object arrays; unwrap if needed.
    if isinstance(pars_list, np.ndarray):
        pars_list = list(pars_list)
    plot_distribution(
        rir_pars=pars_list,
        saveto=os.path.join(args.rir_dir, "data_distribution.png"),
        rir_nums=tuple(args.rir_nums) if args.rir_nums is not None else None,
    )

"""
usage:
 python gen_rir.py --room_size_lims '[[5,5],[6,6],[2.5,2.5]]' --mic_zlim '[2,2]' --spk_zlim '[2,2]' --RT60_lim '[0.2,1.5]' --arr_room_center_dist 20 --spk_arr_dist '[0.3,3]' --rir_nums '[100000,5000,5000]' --fs 16000 --rir_dir /storage/reie/data/rec-rir/rir-w-localization  
 python gen_rir.py --room_size_lims '[[3,15],[3,15],[2.5,6]]' --mic_zlim '[0.5,2]' --spk_zlim '[0.5,2]' --RT60_lim '[0.2,1.5]' --arr_room_center_dist 20 --spk_arr_dist '[0.3,3]' --rir_nums '[100000,5000,5000]' --fs 16000 --rir_dir /mnt/inspurfs/home/wangpengyu/N-RKEM/v5/NeGI/data/sim_rir_final  
  python /mnt/c/Users/reiem/PythonProjects/Rec-RIR/dataset/gen_rir.py --room_size_lims '[[3,15],[3,15],[2.5,6]]' --mic_zlim '[0.5,2]' --spk_zlim '[0.5,2]' --RT60_lim '[0.2,1.5]' --arr_room_center_dist 20 --spk_arr_dist '[0.3,3]' --rir_nums '[100,50, 50]' --fs 16000 --rir_dir /mnt/c/Users/reiem/PythonProjects/Rec-RIR/data/rirs

 Fixed mic center near a wall + restrict sources to inward half-space (nearest wall):
 python gen_rir.py --room_size_lims '[[5,5],[6,6],[2.5,2.5]]' --mic_zlim '[2,2]' --spk_zlim '[2,2]' --RT60_lim '[0.2,1.5]' --mic_center '[1.00,2.00,1.50]' --restrict_src_to_inward_halfspace true --rir_nums '[10000,500,500]' --fs 16000 --rir_dir /storage/reie/data/rec-rir/rir-multiple-rooms

 Fixed room size list + matching mic centers (evenly distributed by index):
 python gen_rir.py --room_size_list '[[5,6,3],[6,7,3],[8,5,3]]' --mic_center_list '[[2.00,1.00,2.0],[2.00,1.00,2.0],[2.00,1.00,2.0]]' --mic_zlim '[1,2]' --spk_zlim '[1,2]' --RT60_lim '[0.2,1.5]' --rir_nums '[10000,500,500]' --fs 16000 --rir_dir /storage/reie/data/rec-rir/rir-fixed-room-list

 Fixed mic center by room dimension ratios (random room sizes):
 python gen_rir.py --room_size_lims '[[3,15],[3,15],[2.5,6]]' --mic_center_ratio '[0.35,0.25,0.6]' --mic_zlim '[1,2]' --spk_zlim '[1,2]' --RT60_lim '[0.2,1.5]' --rir_nums '[10000,500,500]' --fs 16000 --rir_dir /storage/reie/data/rec-rir/rir-ratio-center-multiple-rooms

 Uniform (angle, radius) distribution with ratio-placed mic. r_hi=3.0 is the
 largest radius at which ~96% of the targeted (angle_bin, radius_bin) cells
 are geometrically reachable in rooms [3,15]^2 with mic_center_ratio
 [0.35, 0.25, 0.6]; any larger r_hi relies on phase-2 relaxation for narrow
 angles and re-introduces a radius skew.
 python gen_rir.py --room_size_lims '[[3,15],[3,15],[2.5,6]]' --mic_center_ratio '[0.35,0.25,0.6]' --mic_zlim '[1,2]' --spk_zlim '[1,2]' --RT60_lim '[0.2,1.5]' --spk_arr_dist '[0.3,3.0]' --restrict_src_to_inward_halfspace true --uniform_angle_radius true --rir_nums '[100000,5000,5000]' --fs 16000 --rir_dir /storage/reie/data/rec-rir/rir-uniform-angle-radius

 Uniform angle + full-room radius (no hard r_hi cap). spk_arr_dist[1]=null
 lets each source reach anywhere in the room from the mic at the target angle.
 The angle marginal is kept exactly flat by the theta round-robin; the radius
 marginal spans [0.3, r_max(theta)] where r_max varies with room size and angle.
 python gen_rir.py --room_size_lims '[[3,15],[3,15],[2.5,6]]' --mic_center_ratio '[0.35,0.25,0.6]' --mic_zlim '[1,2]' --spk_zlim '[1,2]' --RT60_lim '[0.2,1.5]' --spk_arr_dist '[0.3,null]' --restrict_src_to_inward_halfspace true --uniform_angle_radius true --rir_nums '[100000,5000,5000]' --fs 16000 --rir_dir /storage/reie/data/rec-rir/rir-uniform-angle-fullroom
 # Or use the config file:
 python gen_rir.py -c /storage/reie/data/rec-rir/rir-uniform-angle-fullroom/config.json
"""
