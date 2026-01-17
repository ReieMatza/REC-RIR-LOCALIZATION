# @author: Pengyu Wang
# @email: wangpengyu@westlake.edu.cn
# @description: Evaluation script for RecRIR model

from jsonargparse import ArgumentParser
import os
import toml
import torch
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import torchaudio

from trainer_inferencer.utils import initialize_module
from model.lossF import RIMag_loss, MSE_loss_complex
import torch.nn.functional as F
from estimate_T60_DRR import cal_T60, cal_DRR

import warnings
warnings.filterwarnings("ignore")


# ============================================================================
# MultiResolution STFT Loss Classes (from Acoustic_Localization_one_mic)
# ============================================================================
def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, fft_size // 2 + 1, #Frames).
    """
    x_stft = torch.stft(x.float(), fft_size, hop_size, win_length, window, return_complex=True)
    x_mag = torch.sqrt(torch.clamp((x_stft.real**2) + (x_stft.imag**2), min=1e-8))
    return x_mag


class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #freq_bins, #frames).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #freq_bins, #frames).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag.float() - x_mag.float(), p="fro") / (torch.norm(y_mag.float(), p="fro") + 1e-12)


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B,  #freq_bins, #frames).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B,  #freq_bins, #frames).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(y_mag.float()), torch.log(x_mag.float()))


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(
        self,
        fft_size=1024,
        shift_size=120,
        win_length=600,
        window="hann_window",
    ):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
        log_mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)
        return sc_loss, log_mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(
        self,
        fft_sizes=[64, 512, 2048, 8192],
        hop_sizes=[32, 256, 1024, 4096],
        win_lengths=[64, 512, 2048, 8192],
        window="hann_window",
        sc_weight=1.0,
        mag_weight=1.0,
    ):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
            factor (float): a balancing factor across different losses.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        self.fft_sizes = fft_sizes
        self.sc_weight = sc_weight
        self.mag_weight = mag_weight

        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses = self.stft_losses + [STFTLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, 1, T).
            y (Tensor): Groundtruth signal (B, 1, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        x = x.squeeze(1)
        y = y.squeeze(1)

        for i, f in enumerate(self.stft_losses):
            sc_l, mag_l = f(x, y)
            sc_loss = sc_loss + sc_l
            mag_loss = mag_loss + mag_l

        return {
            "total": (sc_loss * self.sc_weight + mag_loss * self.mag_weight) / len(self.stft_losses),
            "sc_loss": sc_loss / len(self.stft_losses),
            "mag_loss": mag_loss / len(self.stft_losses),
        }

# ============================================================================


def evaluate(config_path, checkpoint_path, num_samples, device):
    """
    Evaluate the model on validation dataset
    
    Args:
        config_path: Path to config TOML file
        checkpoint_path: Path to model checkpoint
        num_samples: Number of samples to evaluate
        device: Device to use for evaluation
    """
    
    # Load configuration
    config_path = Path(config_path).expanduser().absolute()
    config = toml.load(config_path.as_posix())
    
    # Set seed for reproducibility
    seed = config["meta"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Setup device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Initialize acoustic transform
    acoustic_config = config["acoustic"]
    transformfunc = initialize_module(
        acoustic_config["path"], acoustic_config["args"]
    )
    
    # Initialize model
    print("Loading model...")
    model = initialize_module(config["model"]["path"], args=config["model"]["args"])
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint_path = Path(checkpoint_path).expanduser().absolute()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    # Handle checkpoint state dict (remove DDP 'module.' prefix if present and filter ops/params)
    state_dict = ckpt["model"]
    state_dict = {
        k.replace("module.", ""): v
        for k, v in state_dict.items()
        if not any(x in k for x in ["ops", "params"])
    }
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    # Setup dataloader (single GPU, no distributed)
    # Force evaluation SNR to 20 dB without touching training configs
    config["dataloader"]["args"]["snr_range"] = [20, 20]
    print("Loading validation dataset...")
    config["dataloader"]["args"].update({"rank": 0})
    config["dataloader"]["args"].update({"sr": config["acoustic"]["args"]["sr"]})
    
    dataloader = initialize_module(
        config["dataloader"]["path"], args=config["dataloader"]["args"]
    )
    valid_dataloader = dataloader.valid_dataloader
    
    # Setup loss function
    loss_w_rvb = config["trainer"]["train"]["loss_w_rvb"]
    loss_w_cln = config["trainer"]["train"]["loss_w_cln"]
    loss_w_rec = config["trainer"]["train"]["loss_w_rec"]
    loss_type = config["trainer"]["train"]["loss_type"]
    
    if loss_type == "RIMag":
        lossF = RIMag_loss()
    elif loss_type == "MSE":
        lossF = MSE_loss_complex()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    print(f"Using loss type: {loss_type}")
    print(f"Loss weights - Clean: {loss_w_cln}, Reverb: {loss_w_rvb}, Recon: {loss_w_rec}")
    
    # Initialize PIM module for RIR estimation
    print("Initializing PIM module for RIR estimation...")
    pim = initialize_module(config["EM_algo"]["path"], config["EM_algo"]["args"])
    
    # Initialize MultiResolution STFT Loss for RIR evaluation
    print("Initializing MultiResolution STFT Loss...")
    mrstft_loss = MultiResolutionSTFTLoss(
        fft_sizes=[64, 512, 2048, 8192],
        hop_sizes=[32, 256, 1024, 4096],
        win_lengths=[64, 512, 2048, 8192],
        window="hann_window",
        sc_weight=1.0,
        mag_weight=1.0,
    ).to(device)
    
    # Access ground truth RIR paths from validation dataset
    valid_dataset = valid_dataloader.dataset
    rir_pathlist = valid_dataset.rir_pathlist
    print(f"Found {len(rir_pathlist)} RIR files in validation dataset")
    
    # Evaluation loop
    print(f"\nEvaluating on {num_samples} samples...")
    loss_total = 0.0
    loss_total_cln = 0.0
    loss_total_rvb = 0.0
    loss_total_rec = 0.0
    loss_mrstft_total = 0.0
    loss_mrstft_sc = 0.0
    loss_mrstft_mag = 0.0
    
    # RT60 and DRR metric accumulators
    rt60_mae_total = 0.0
    rt60_rmse_total = 0.0
    drr_mae_total = 0.0
    drr_rmse_total = 0.0
    acoustic_metrics_count = 0
    
    samples_evaluated = 0
    
    # Create progress bar that tracks samples, not batches
    pbar = tqdm(total=num_samples, desc="Evaluating", unit="sample")
    
    with torch.no_grad():
        for index, (
            noisy_wav,
            rev_wav,
            dp_wav,
            fpath,
            rir_path,
            _angle_class,
            _radius_class,
        ) in enumerate(valid_dataloader):
            if samples_evaluated >= num_samples:
                break
            
            batch_size = noisy_wav.shape[0]
            # Adjust batch size if we're near the limit
            remaining_samples = num_samples - samples_evaluated
            if remaining_samples < batch_size:
                # Only process the samples we need
                noisy_wav = noisy_wav[:remaining_samples]
                rev_wav = rev_wav[:remaining_samples]
                dp_wav = dp_wav[:remaining_samples]
                batch_size = remaining_samples
            
            # Move data to device
            noisy_wav = noisy_wav.to(device)
            rev_wav = rev_wav.to(device)
            dp_wav = dp_wav.to(device)
            
            # Apply STFT transforms
            input_complex = transformfunc.stft(noisy_wav, output_type="complex")
            target_complex = transformfunc.stft(dp_wav, output_type="complex")
            reverb_complex = transformfunc.stft(rev_wav, output_type="complex")
            
            # Preprocess for model input
            input_ft = transformfunc.preprocess(input_complex)
            
            # Forward pass
            (
                est_spch_ft,
                est_ctf_ft,
                est_reverb_ft,
                _angle_logits,
                _radius_logits,
            ) = model(input_ft)
            
            # Postprocess model outputs
            est_spch = transformfunc.postprocess(est_spch_ft)
            est_ctf = transformfunc.postprocess(est_ctf_ft)
            est_reverb = transformfunc.postprocess(est_reverb_ft)
            
            # Calculate reconstruction via convolution with CTF
            recon = torchaudio.functional.convolve(
                target_complex, est_ctf, mode="full"
            )[..., : target_complex.shape[-1]]
            
            # Calculate losses
            loss_cln = lossF(est_spch, target_complex)
            loss_rvb = lossF(est_reverb, reverb_complex)
            loss_rec = lossF(recon, reverb_complex)
            
            loss = (
                loss_w_cln * loss_cln
                + loss_w_rec * loss_rec
                + loss_w_rvb * loss_rvb
            )
            
            # ===== RIR Quality Evaluation using MRSTFT Loss =====
            # Use per-sample RIR path provided by dataset
            
            # Load ground truth RIR
            try:
                if isinstance(rir_path, (list, tuple)):
                    rir_paths = list(rir_path)
                else:
                    rir_paths = [rir_path] * batch_size

                if len(rir_paths) != batch_size:
                    rir_paths = (rir_paths * batch_size)[:batch_size]

                gt_rirs = []
                for rp in rir_paths:
                    # Check if it's .npz format (contains both rir and rir_dp)
                    if rp.endswith('.npz'):
                        rir_dict = np.load(rp)
                        gt_rir = torch.from_numpy(rir_dict["rir"][0]).float().to(device)
                        # Normalize
                        scale_rir = gt_rir.abs().max()
                        if scale_rir > 0:
                            gt_rir = gt_rir / scale_rir
                    else:
                        # Load from wav file
                        gt_rir = transformfunc.load_wav(rp, transformfunc.sr).to(device)
                        # Normalize
                        if gt_rir.abs().max() > 0:
                            gt_rir = gt_rir / gt_rir.abs().max()
                    gt_rirs.append(gt_rir.flatten())
                
                # Estimate RIR from CTF using PIM sine sweep method
                # Flip CTF as done in PIM.init_seg (line 71)
                est_ctf_flipped = est_ctf.flip(-1)  # [B, C, F, L]
                batch_size = est_ctf_flipped.shape[0]
                
                # Use refactored PIM method to convert CTF to RIR (handles batching internally)
                # Keep on device for loss calculation
                est_rirs = pim.ctf_to_rir(est_ctf_flipped, transformfunc, device, return_device=device)
                
                # Pad/truncate to same length for loss calculation
                max_len = max(
                    max(r.shape[-1] for r in est_rirs),
                    max(r.shape[-1] for r in gt_rirs),
                )
                gt_rir_batch = torch.stack([
                    torch.nn.functional.pad(r, (0, max_len - r.shape[-1]))
                    for r in gt_rirs
                ])  # [B, T]
                est_rir_batch = torch.stack([
                    torch.nn.functional.pad(r, (0, max_len - r.shape[-1]))
                    for r in est_rirs
                ])  # [B, T]
                
                # Calculate MultiResolution STFT Loss
                # MRSTFT expects shape [B, 1, T]
                mrstft_result = mrstft_loss(
                    est_rir_batch.unsqueeze(1),  # [B, 1, T]
                    gt_rir_batch.unsqueeze(1)  # [B, 1, T]
                )
                
                # Accumulate MRSTFT losses
                loss_mrstft_total += mrstft_result["total"].item() * batch_size
                loss_mrstft_sc += mrstft_result["sc_loss"].item() * batch_size
                loss_mrstft_mag += mrstft_result["mag_loss"].item() * batch_size
                
                # ===== Calculate RT60 and DRR metrics =====
                # Calculate RT60 and DRR for each sample in batch
                for b in range(batch_size):
                    # Get estimated and ground truth RIRs
                    est_rir_sample = est_rirs[b]
                    gt_rir_sample = gt_rirs[b]
                    
                    # Calculate RT60 using the original paper's implementation
                    rt60_est = cal_T60(
                        est_rir_sample.detach().cpu().numpy(),
                        sr=transformfunc.sr,
                        savepath=None,
                    )
                    rt60_gt = cal_T60(
                        gt_rir_sample.detach().cpu().numpy(),
                        sr=transformfunc.sr,
                        savepath=None,
                    )
                    
                    # Calculate DRR using the original paper's implementation
                    drr_est = cal_DRR(est_rir_sample.detach().cpu(), sr=transformfunc.sr)
                    drr_gt = cal_DRR(gt_rir_sample.detach().cpu(), sr=transformfunc.sr)
                    
                    # Calculate errors (MAE and squared error for RMSE)
                    rt60_error = abs(rt60_est - rt60_gt)
                    rt60_squared_error = (rt60_est - rt60_gt) ** 2
                    drr_error = abs(drr_est - drr_gt)
                    drr_squared_error = (drr_est - drr_gt) ** 2
                    
                    # Accumulate
                    rt60_mae_total += rt60_error
                    rt60_rmse_total += rt60_squared_error
                    drr_mae_total += drr_error
                    drr_rmse_total += drr_squared_error
                    acoustic_metrics_count += 1
                        

                
            except Exception as e:
                print(f"\nWarning: Could not compute MRSTFT loss for batch {index}: {e}")
                # Continue without adding to MRSTFT loss
            
            # Accumulate losses
            loss_total += loss.item()
            loss_total_cln += loss_cln.item()
            loss_total_rvb += loss_rvb.item()
            loss_total_rec += loss_rec.item()
            
            samples_evaluated += batch_size
            pbar.update(batch_size)
    
    pbar.close()
    
    # Calculate average losses
    avg_loss_total = loss_total / samples_evaluated
    avg_loss_cln = loss_total_cln / samples_evaluated
    avg_loss_rvb = loss_total_rvb / samples_evaluated
    avg_loss_rec = loss_total_rec / samples_evaluated
    avg_mrstft_total = loss_mrstft_total / samples_evaluated
    avg_mrstft_sc = loss_mrstft_sc / samples_evaluated
    avg_mrstft_mag = loss_mrstft_mag / samples_evaluated
    
    # Calculate average RT60 and DRR metrics
    if acoustic_metrics_count > 0:
        avg_rt60_mae = rt60_mae_total / acoustic_metrics_count
        avg_rt60_rmse = np.sqrt(rt60_rmse_total / acoustic_metrics_count)
        avg_drr_mae = drr_mae_total / acoustic_metrics_count
        avg_drr_rmse = np.sqrt(drr_rmse_total / acoustic_metrics_count)
    else:
        avg_rt60_mae = avg_rt60_rmse = avg_drr_mae = avg_drr_rmse = 0.0
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Evaluation Results ({samples_evaluated} samples):")
    print(f"{'='*50}")
    print(f"Total Loss:     {avg_loss_total:.6f}")
    print(f"Clean Loss:     {avg_loss_cln:.6f}")
    print(f"Reverb Loss:    {avg_loss_rvb:.6f}")
    print(f"Recon Loss:     {avg_loss_rec:.6f}")
    print(f"{'-'*50}")
    print(f"RIR Quality (MultiResolution STFT):")
    print(f"  Total:        {avg_mrstft_total:.6f}")
    print(f"  Spectral Conv:{avg_mrstft_sc:.6f}")
    print(f"  Log Magnitude:{avg_mrstft_mag:.6f}")
    print(f"{'-'*50}")
    print(f"Acoustic Metrics (RT60 & DRR):")
    print(f"  RT60 MAE:     {avg_rt60_mae:.6f} s")
    print(f"  RT60 RMSE:    {avg_rt60_rmse:.6f} s")
    print(f"  DRR MAE:      {avg_drr_mae:.6f} dB")
    print(f"  DRR RMSE:     {avg_drr_rmse:.6f} dB")
    print(f"{'='*50}\n")

    # CSV-friendly output for quick copy/paste into spreadsheets
    print(
        "csv,total_loss,clean_loss,reverb_loss,recon_loss,"
        "mrstft_total,mrstft_sc,mrstft_mag,"
        "rt60_mae_s,rt60_rmse_s,drr_mae_db,drr_rmse_db"
    )
    print(
        "csv,"
        f"{avg_loss_total:.6f},{avg_loss_cln:.6f},{avg_loss_rvb:.6f},{avg_loss_rec:.6f},"
        f"{avg_mrstft_total:.6f},{avg_mrstft_sc:.6f},{avg_mrstft_mag:.6f},"
        f"{avg_rt60_mae:.6f},{avg_rt60_rmse:.6f},{avg_drr_mae:.6f},{avg_drr_rmse:.6f}"
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="RecRIR Model Evaluation")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config/Rec-RIR.toml",
        help="Path to config TOML file (default: config/Rec-RIR.toml)"
    )
    parser.add_argument(
        "-k", "--checkpoint",
        default="saved_dirpath2/ckpt/best.tar",
        type=str,
        help="Path to model checkpoint (required)"
    )
    parser.add_argument(
        "-n", "--num_samples",
        type=int,
        default=50,
        help="Number of samples to evaluate (default: 20)"
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use: 'cpu', 'cuda', or 'cuda:N' (default: cuda if available)"
    )
    
    args = parser.parse_args()
    
    evaluate(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        device=args.device
    )
