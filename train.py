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

    # Freeze backbone, only train localization heads
    freeze_backbone = config["meta"].get("freeze_backbone", False)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.angle_head.parameters():
            param.requires_grad = True
        for param in model.radius_head.parameters():
            param.requires_grad = True
        if hasattr(model, "film_conditioner") and model.film_conditioner is not None:
            for param in model.film_conditioner.parameters():
                param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Freeze backbone: training {trainable}/{total} parameters "
              f"({100*trainable/total:.1f}%)")

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
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze all parameters except angle_head and radius_head",
    )

    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])

    config_path = Path(args.config).expanduser().absolute()

    config = toml.load(config_path.as_posix())

    config["meta"]["config_path"] = args.config
    config["meta"]["save_dir"] = args.save_path
    config["meta"]["start_ckpt"] = args.start_ckpt
    config["meta"]["freeze_backbone"] = args.freeze_backbone

    entry(local_rank, config, args.resume, args.start_ckpt)
    """
    usage: torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py -c /storage/reie/REC-RIR-LOCALIZATION/config/Rec-RIR-quick.toml -p /storage/reie/experiments/w-localization-fixed-room-list
    """
