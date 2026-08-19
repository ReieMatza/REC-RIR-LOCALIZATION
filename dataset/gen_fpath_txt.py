# @author: Pengyu Wang
# @email: wangpengyu@westlake.edu.cn
# @description: code for save filepath list to .txt.

from glob import glob
import os
from jsonargparse import ArgumentParser


def save_addresses_to_txt(addresses: list, out_path: str):
    with open(out_path, "w") as file:
        for address in addresses:
            file.write(address + "\n")

    return None


def gen_txt(dir_path: str, filetype: str):

    target_path = sorted(
        glob(os.path.join(dir_path, "**", "*{}".format(filetype)), recursive=True)
    )

    return target_path


def main(input_dir, output_path, ext):

    addresses = gen_txt(input_dir, ext)
    save_addresses_to_txt(addresses, output_path)

    return None


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--input_dir", required=True, type=str, help="input dirpath"
    )
    parser.add_argument(
        "-o", "--output_path", required=True, type=str, help="output filepath"
    )
    parser.add_argument(
        "-e", "--ext", required=True, type=str, help="filename extension"
    )
    args = parser.parse_args()
    main(**args)

    """
    python gen_fpath_txt.py -i /storage/reie/data/rec-rir/rir-fixed-room-list/train -o /storage/reie/REC-RIR-LOCALIZATION/config/rir-fixed-room-list/reie-train-rir.txt -e .npz
    python gen_fpath_txt.py -i /storage/reie/data/rec-rir/rir-fixed-room-list/validation -o /storage/reie/REC-RIR-LOCALIZATION/config/rir-fixed-room-list/reie-validation-rir.txt -e .npz
    python gen_fpath_txt.py -i /storage/reie/data/rec-rir/rir-fixed-room-list/test -o /storage/reie/REC-RIR-LOCALIZATION/config/rir-fixed-room-list/reie-test-rir.txt -e .npz



    python gen_fpath_txt.py -i /storage/reie/data/rec-rir/rirs/as_in_paper/train -o /storage/reie/REC-RIR-LOCALIZATION/config/as_in_paper/reie-train-rir.txt -e .npz
    python gen_fpath_txt.py -i /storage/reie/data/rec-rir/rirs/as_in_paper/validation -o /storage/reie/REC-RIR-LOCALIZATION/config/as_in_paper/reie-validation-rir.txt -e .npz
    python gen_fpath_txt.py -i /storage/reie/data/rec-rir/rirs/as_in_paper/test -o /storage/reie/REC-RIR-LOCALIZATION/config/as_in_paper/reie-test-rir.txt -e .npz

    python gen_fpath_txt.py -i /storage/reie/data/rec-rir/rir-ratio-center-multiple-rooms/train -o /storage/reie/REC-RIR-LOCALIZATION/config/rir-ratio-center-multiple-rooms/reie-train-rir.txt -e .npz
    python gen_fpath_txt.py -i /storage/reie/data/rec-rir/rir-ratio-center-multiple-rooms/validation -o /storage/reie/REC-RIR-LOCALIZATION/config/rir-ratio-center-multiple-rooms/reie-validation-rir.txt -e .npz
    python gen_fpath_txt.py -i /storage/reie/data/rec-rir/rir-ratio-center-multiple-rooms/test -o /storage/reie/REC-RIR-LOCALIZATION/config/rir-ratio-center-multiple-rooms/reie-test-rir.txt -e .npz

    python gen_fpath_txt.py -i /storage/reie/data/rec-rir/rir-uniform-angle-fullroom-edge/train -o /storage/reie/REC-RIR-LOCALIZATION/config/rir-uniform-angle-fullroom-edge/reie-train-rir.txt -e .npz
    python gen_fpath_txt.py -i /storage/reie/data/rec-rir/rir-uniform-angle-fullroom-edge/validation -o /storage/reie/REC-RIR-LOCALIZATION/config/rir-uniform-angle-fullroom-edge/reie-validation-rir.txt -e .npz
    python gen_fpath_txt.py -i /storage/reie/data/rec-rir/rir-uniform-angle-fullroom-edge/test -o /storage/reie/REC-RIR-LOCALIZATION/config/rir-uniform-angle-fullroom-edge/reie-test-rir.txt -e .npz
    """
    
