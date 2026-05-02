# @author: Pengyu Wang
# @email: wangpengyu@westlake.edu.cn
# @description: trainer base class.

import os
from os import path
import json
import numbers
import toml
import time
import logging
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from .utils import initialize_module


def _to_jsonable_scalar(value):
    """Coerce a wandb.log payload value to a plain JSON scalar, or return None
    if the value isn't a scalar we can safely persist (e.g. wandb.Image).

    Handles: Python numbers, bool, torch.Tensor (0-dim or 1-element), numpy
    scalars. Anything else (including lists/dicts/images) is dropped from the
    local JSONL mirror.
    """
    if isinstance(value, bool):  # bool is a numbers.Number, keep it first.
        return value
    if isinstance(value, numbers.Number):
        return float(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().item())
        return None
    try:
        import numpy as _np  # local import to avoid unconditional dependency
        if isinstance(value, _np.generic):
            return float(value)
    except Exception:
        pass
    return None


def _filter_shape_mismatches(ckpt_state, model_state, logger: bool = False):
    """Drop tensors from ckpt_state whose shape disagrees with model_state.

    Needed because torch's load_state_dict(strict=False) ignores missing /
    unexpected KEYS but still raises on SHAPE mismatches for present keys.
    This path is hit whenever a head's output dim changes between runs
    (e.g. num_rad_classes 31 -> 49 after widening the radius range).
    Shape-mismatched layers fall back to their freshly-initialized weights.
    """
    filtered = {}
    dropped = []
    for k, v in ckpt_state.items():
        if k in model_state and v.shape != model_state[k].shape:
            dropped.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            continue
        filtered[k] = v
    if logger and dropped:
        print(
            f"[start_ckpt] dropped {len(dropped)} tensor(s) due to shape mismatch "
            f"(these layers keep their fresh init):"
        )
        for name, ckpt_shape, model_shape in dropped:
            print(f"  - {name}: ckpt {ckpt_shape} != model {model_shape}")
    return filtered


class BaseTrainer:

    def __init__(
        self,
        dist,
        rank,
        config,
        resume: bool,
        model,
        optimizer,
        scheduler,
        start_ckpt: None,
    ):

        self.dist = dist
        self.rank = rank

        self.model = DDP(
            model.cuda(rank), device_ids=[rank], find_unused_parameters=False
        )
        
        # Conditionally compile model based on config
        use_compile = config["meta"].get("use_torch_compile", True)
        if use_compile:
            self.compiled_model = torch.compile(self.model)
            if rank == 0:
                print("Using torch.compile for model optimization")
        else:
            self.compiled_model = self.model
            if rank == 0:
                print("torch.compile disabled - using standard model")

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_dataloader = None

        num_gpus = int(os.environ["WORLD_SIZE"])
        config["dataloader"]["args"]["batchsize"][0] *= num_gpus
        config["dataloader"]["args"]["batchsize"][1] *= num_gpus

        # meta config
        self.meta_config = config["meta"]
        torch.backends.cudnn.enabled = self.meta_config["cudnn_enable"]
        self.save_dir = self.meta_config["save_dir"]
        self.ckpt_dir = path.join(self.save_dir, "ckpt")
        self.log_dir = path.join(self.save_dir, "log")

        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # acoustic config
        self.acoustic_config = config["acoustic"]
        self.transformfunc = initialize_module(
            self.acoustic_config["path"], self.acoustic_config["args"]
        )
        self.sr = self.transformfunc.sr

        # training config
        self.train_config = config["trainer"]["train"]
        self.epochs = self.train_config["epochs"]
        self.save_ckpt_interval = self.train_config["save_ckpt_interval"]
        self.clip_grad_norm_value = self.train_config["clip_grad_norm_value"]

        # validation config
        self.valid_config = config["trainer"]["validation"]
        self.valid_interval = self.valid_config["interval"]
        self.save_max_metric = self.valid_config["save_max_metric"]

        # logger
        self.logger = logging.getLogger("mylogger")
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(path.join(self.log_dir, "log.txt"), mode="a")
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        # Local metrics mirror (JSONL). Rank 0 only. Each `_log_metrics` call
        # appends a single line:
        #     {"step": <step_or_null>, "<metric_name>": <scalar>, ...}
        # Non-scalar values (wandb.Image, etc.) are dropped from the mirror.
        # Kept as an open file handle with line-buffered flushes so a crash or
        # SIGKILL still preserves everything logged up to the last call.
        self._metrics_jsonl_path = path.join(self.log_dir, "metrics.jsonl")
        if self.rank == 0:
            # Append so that `--resume` appends to the same file across restarts.
            self._metrics_jsonl_fh = open(self._metrics_jsonl_path, "a", buffering=1)
        else:
            self._metrics_jsonl_fh = None

        # initialization
        self.start_epoch = 1
        self.best_metric = -torch.inf if self.save_max_metric else torch.inf
        self.steps = 0
        if resume:
            self._resume_ckpt()

        if start_ckpt:
            ckpt = torch.load(start_ckpt, map_location="cpu")

            self.dist.barrier()
            ckpt["model"] = {
                k: v
                for k, v in ckpt["model"].items()
                if not any(x in k for x in ["ops", "params"])
            }
            # strict=False only ignores missing/unexpected keys; it does NOT
            # handle shape mismatches. If a layer's output dim changed (e.g.
            # radius_head grew from 31 to 49 classes when we widened the radius
            # range), PyTorch raises. Drop shape-mismatched tensors so the rest
            # of the checkpoint still warm-starts and the mismatched layers
            # fall back to their freshly-initialized weights.
            ckpt["model"] = _filter_shape_mismatches(
                ckpt["model"], self.model.state_dict(), logger=self.rank == 0
            )
            self.model.load_state_dict(ckpt["model"], strict=False)

        if self.rank == 0:
            # Initialize WandB with config from TOML
            wandb_config = config.get("wandb", {})            
            wandb.init(
                project=wandb_config.get("project", "rec-rir-localization"),
                tags=wandb_config.get("tags", []),
                notes=wandb_config.get("notes", ""),
                config=config,
                dir=self.log_dir,
                mode=wandb_config.get("mode", "online"),
                group=wandb_config.get("group") or None,
                job_type=wandb_config.get("job_type", "train"),
                resume="allow" if resume else None
            )
            
            with open(
                path.join(self.save_dir, f"{time.strftime('%Y-%m-%d-%H-%M-%S')}.toml"),
                "w",
            ) as handle:
                toml.dump(config, handle)

    def _save_best_latest_ckpt(
        self, epoch: int, is_best: bool = False, period: bool = False
    ):
        torch.cuda.synchronize()
        state_dict = {
            "epoch": epoch,
            "best_metric": self.best_metric,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "steps": self.steps,
        }

        state_dict["model"] = self.model.state_dict()

        torch.save(state_dict, path.join(self.ckpt_dir, f"latest.tar"))
        if is_best:
            self.logger.info(f"New best model saved")
            torch.save(state_dict, path.join(self.ckpt_dir, f"best.tar"))
        if period:
            torch.save(state_dict, path.join(self.ckpt_dir, f"epoch{epoch}.tar"))

    def _is_best(self, metric):
        """
        check if the checkpoint is the best
        """
        if self.save_max_metric and metric > self.best_metric:
            self.best_metric = metric
            return True
        elif not self.save_max_metric and metric < self.best_metric:
            self.best_metric = metric
            return True
        else:
            return False

    def _resume_ckpt(self):
        """resume training"""
        latest_model_path = path.join(self.ckpt_dir, "latest.tar")
        assert path.exists(latest_model_path), f"{latest_model_path} does not exist"

        ckpt = torch.load(latest_model_path, map_location="cpu")

        self.dist.barrier()

        self.start_epoch = int(ckpt["epoch"] + 1)
        self.best_metric = ckpt["best_metric"]
        self.steps = ckpt["steps"]

        ckpt["model"] = {
            k: v
            for k, v in ckpt["model"].items()
            if not any(x in k for x in ["ops", "params"])
        }
        ckpt["model"] = _filter_shape_mismatches(
            ckpt["model"], self.model.state_dict(), logger=self.rank == 0
        )
        self.model.load_state_dict(ckpt["model"], strict=False)

        if self.rank == 0:
            self.logger.info(
                f"Model checkpoint is loaded. Training will begin at epoch {self.start_epoch}."
            )

    def _log_metrics(self, metrics: dict, step=None):
        """Log a metrics dict to both WandB and the local JSONL mirror.

        Drop-in replacement for `wandb.log(metrics, step=step)`. The WandB
        call is preserved unchanged; the JSONL mirror on rank 0 records only
        scalar entries (see _to_jsonable_scalar) so wandb.Image payloads are
        safely ignored. Use this from any trainer method that would otherwise
        call `wandb.log` directly.
        """
        if step is None:
            wandb.log(metrics)
        else:
            wandb.log(metrics, step=step)

        if self._metrics_jsonl_fh is None:
            return  # Non-rank-0 ranks skip the local mirror.

        record = {}
        if step is not None:
            record["step"] = int(step)
        for key, value in metrics.items():
            scalar = _to_jsonable_scalar(value)
            if scalar is not None:
                record[key] = scalar
        # If nothing was scalar (e.g. an image-only payload), skip the line
        # entirely rather than emitting a line that's just {"step": N}.
        if len(record) <= (1 if step is not None else 0):
            return
        self._metrics_jsonl_fh.write(json.dumps(record) + "\n")

    def _set_train_mode(self):
        self.compiled_model.train()
        self.model.train()

    def _set_valid_mode(self):
        torch.cuda.synchronize()
        self.compiled_model.eval()
        self.model.eval()

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _validation_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        metric = torch.inf

        for epoch in range(self.start_epoch, self.epochs + 1):

            self.train_dataloader.sampler.set_epoch(epoch)
            if self.rank == 0:
                self.logger.info(f"{'=' * 5} epoch {epoch} {'=' * 5}")
            self._set_train_mode()
            self._train_epoch(epoch)

            if self.rank == 0:
                self._log_metrics(
                    {"Lr": self.optimizer.param_groups[0]["lr"]},
                    step=self.steps,
                )

            if epoch % self.valid_interval == 0:
                self._set_valid_mode()
                metric = self._validation_epoch(epoch)
                if self.rank == 0:
                    if epoch % self.save_ckpt_interval == 0:
                        self._save_best_latest_ckpt(epoch, is_best=False, period=True)
                    self._save_best_latest_ckpt(epoch, is_best=self._is_best(metric))
