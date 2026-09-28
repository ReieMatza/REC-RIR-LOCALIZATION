# @author: Pengyu Wang
# @email: wangpengyu@westlake.edu.cn
# @description: main code for training.

from jsonargparse import ArgumentParser
import os
import toml
import torch
import torch.distributed as dist
import numpy as np
import random
from pathlib import Path
from trainer_inferencer.utils import initialize_module, set_optimizer

import warnings

warnings.filterwarnings("ignore")


os.environ["NCCL_IB_TIMEOUT"] = "22"


def _freeze_loc_readout(model) -> None:
    """Train the Rec-RIR trunk. Freeze loc heads, the loc embedding, and every FiLM module.

    Real-RIR fine-tune has no angle, radius, or room geometry. FiLM stays at the
    loaded checkpoint values and is not applied (room_params=None).
    """
    frozen_names = []
    for name, param in model.named_parameters():
        lname = name.lower()
        if (
            "film" in lname
            or lname.startswith("angle_fc")
            or lname.startswith("radius_head.")
            or lname.startswith("embedding_1dconv.")
        ):
            param.requires_grad = False
            frozen_names.append(name)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Freeze loc readout: training {trainable}/{total} parameters "
        f"({100 * trainable / total:.1f}%) [encoder, speech/noise/CTF, decoders]. "
        f"Frozen {len(frozen_names)} tensors (angle_fc* + radius_head + "
        f"embedding_1dconv + all FiLM)."
    )


def entry(rank, config, resume, start_ckpt):
    # Seed
    seed = config["meta"]["seed"]
    torch.manual_seed(seed)  # For both CPU and GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(rank)

    num_gpus = int(os.environ["WORLD_SIZE"])
    config["dataloader"]["args"]["batchsize"][0] //= num_gpus
    config["dataloader"]["args"]["batchsize"][1] //= num_gpus

    # DDP
    torch.distributed.init_process_group(backend="nccl")
    print(f"Process {rank + 1} initialized.")

    # dataset
    config["dataloader"]["args"].update({"rank": rank})
    config["dataloader"]["args"].update({"sr": config["acoustic"]["args"]["sr"]})

    model = initialize_module(config["model"]["path"], args=config["model"]["args"])

    if config["meta"].get("freeze_loc_readout", False):
        _freeze_loc_readout(model)

    dataloader = initialize_module(
        config["dataloader"]["path"], args=config["dataloader"]["args"]
    )

    optimizer, scheduler = set_optimizer(
        model, config["optimizer"], config["scheduler"]
    )

    trainer_class = initialize_module(config["trainer"]["path"], initialize=False)
    trainer = trainer_class(
        dist=dist,
        rank=rank,
        config=config,
        resume=resume,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=dataloader.train_dataloader,
        valid_dataloader=dataloader.valid_dataloader,
        start_ckpt=start_ckpt,
    )

    trainer.train()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    parser = ArgumentParser(description="VINP2 training")
    parser.add_argument(
        "-c", "--config", required=True, type=str, help="Config .toml file"
    )
    parser.add_argument(
        "-p", "--save_path", required=True, type=str, help="Save folder path"
    )
    parser.add_argument("-r", "--resume", action="store_true", help="Resume training")
    parser.add_argument("-s", "--start_ckpt", type=str, help="start_ckpt", default=None)

    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])

    config_path = Path(args.config).expanduser().absolute()

    config = toml.load(config_path.as_posix())

    config["meta"]["config_path"] = args.config
    config["meta"]["save_dir"] = args.save_path
    config["meta"]["start_ckpt"] = args.start_ckpt

    entry(local_rank, config, args.resume, args.start_ckpt)
    """
    usage: torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py -c config/Rec-RIR-heads-only-epoch35-straight-to-15m-2gpu.toml -p /storage/reie/experiments/epoch35-straight-to-15m-2gpu -s ckpt/epoch35.tar
    """
