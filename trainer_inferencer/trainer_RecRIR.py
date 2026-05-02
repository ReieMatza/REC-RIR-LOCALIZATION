import torch
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
from .base_trainer_RecRIR import BaseTrainer
from .utils import plot_spectrogram
import torchaudio
from model.lossF import RIMag_loss, MSE_loss_complex, GaussianLabelSmoothingLoss
from torch.amp import GradScaler, autocast
import wandb
import torch.nn as nn

lossF_RIMag = RIMag_loss()
lossF_MSE = MSE_loss_complex()


class Trainer(BaseTrainer):
    def __init__(
        self,
        dist,
        rank,
        config,
        resume: bool,
        model,
        optimizer,
        scheduler,
        train_dataloader,
        valid_dataloader,
        start_ckpt,
        *args,
        **kwargs,
    ):
        super().__init__(
            dist,
            rank,
            config,
            resume,
            model,
            optimizer,
            scheduler,
            start_ckpt,
        )

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        train_cfg = config["trainer"]["train"]
        self.loss_w_rvb = train_cfg["loss_w_rvb"]
        self.loss_w_cln = train_cfg["loss_w_cln"]
        self.loss_w_rec = train_cfg["loss_w_rec"]
        self.loss_w_angle = train_cfg.get("loss_w_angle", 0.0)
        self.loss_w_radius = train_cfg.get("loss_w_radius", 0.0)

        self.scaler = GradScaler()
        if train_cfg["loss_type"] == "RIMag":
            self.lossF = lossF_RIMag
        elif train_cfg["loss_type"] == "MSE":
            self.lossF = lossF_MSE

        # Radius class index -> meters conversion (for correct error reporting in meters).
        self.rad_resolution = float(
            config["dataloader"]["args"].get("rad_resolution", 0.1)
        )

        # Classification heads: angle and radius.
        #
        # By default we use hard cross-entropy. If angle_label_smoothing_sigma > 0
        # (or its radius counterpart) we switch to a Gaussian soft target with that
        # sigma in class-index units. See GaussianLabelSmoothingLoss for details.
        #
        # Rationale: both the angle axis (1 deg/class) and radius axis (0.25 m/class
        # after the range fix) are ordinal, and a Dirac one-hot target wastes the
        # network's capacity trying to spike at a bin the task cannot physically
        # resolve (true for single-mic DOA in particular).
        model_args = config["model"]["args"]
        self.num_angle_classes = int(model_args.get("num_angles", 181))
        dl_args = config["dataloader"]["args"]
        max_rad_value = float(
            model_args.get("max_rad_value", dl_args.get("max_rad_value", 3.0))
        )
        self.num_radius_classes = int(max_rad_value / self.rad_resolution) + 1

        angle_sigma = float(train_cfg.get("angle_label_smoothing_sigma", 0.0))
        radius_sigma = float(train_cfg.get("radius_label_smoothing_sigma", 0.0))
        self.angle_label_smoothing_sigma = angle_sigma
        self.radius_label_smoothing_sigma = radius_sigma

        if angle_sigma > 0:
            self.angle_loss = GaussianLabelSmoothingLoss(
                num_classes=self.num_angle_classes, sigma=angle_sigma
            ).to(self.rank)
        else:
            self.angle_loss = nn.CrossEntropyLoss()

        if radius_sigma > 0:
            self.radius_loss = GaussianLabelSmoothingLoss(
                num_classes=self.num_radius_classes, sigma=radius_sigma
            ).to(self.rank)
        else:
            self.radius_loss = nn.CrossEntropyLoss()

        # Keep classification_loss for any external code / checkpoints that still
        # reference it, but the actual losses above are what the train/val loops use.
        self.classification_loss = nn.CrossEntropyLoss()

        # Which scalar the base trainer uses to decide "best" checkpoint.
        # Allowed values (all lower-is-better; keep save_max_metric=false):
        #   "total"          : loss_cln + loss_rvb + loss_rec + loss_angle + loss_radius (weighted).
        #                      Legacy default; dominated by the STFT regression terms.
        #   "angle_ce"       : angle-head loss in KL form (CE minus target entropy).
        #                      With hard CE (sigma=0) this equals raw CE; with Gaussian
        #                      label smoothing it zeros out at a perfect predictor
        #                      regardless of sigma, so runs with different sigma are
        #                      comparable.
        #   "radius_ce"      : radius-head loss in KL form (same convention).
        #   "localization_ce": angle_ce + radius_ce (both KL).
        #   "angle_error"    : mean absolute angle error in degrees (argmax).
        #   "radius_error"   : mean absolute radius error in meters (argmax, converted via rad_resolution).
        #   "localization"   : angle_error_deg + best_metric_radius_weight * radius_error_m.
        # NOTE: switching this mid-experiment invalidates any saved best_metric in latest.tar
        # because the old value is in different units -- start a fresh experiment dir when changing.
        validation_cfg = config["trainer"].get("validation", {})
        self.best_metric_name = validation_cfg.get("best_metric", "total")
        allowed = {
            "total",
            "angle_ce",
            "radius_ce",
            "localization_ce",
            "angle_error",
            "radius_error",
            "localization",
        }
        if self.best_metric_name not in allowed:
            raise ValueError(
                f"[trainer.validation] best_metric='{self.best_metric_name}' is not one of {sorted(allowed)}"
            )
        # Weight on the radius term in the "localization" combo metric. Units: deg/m.
        # Default 10 deg/m makes a 1 m radius error roughly as bad as a 10 deg angle error.
        self.best_metric_radius_weight = float(
            validation_cfg.get("best_metric_radius_weight", 10.0)
        )

    def _train_epoch(self, epoch):

        loss_total = 0.0
        self.optimizer.zero_grad()
        for index, (noisy_wav, rev_wav, dp_wav, fpath, rir_path, angle_class, radius_class, room_params) in (
            enumerate(tqdm(self.train_dataloader, desc="Training"))
            if self.rank == 0
            else enumerate(self.train_dataloader)
        ):
            self.optimizer.zero_grad()
            with autocast(dtype=torch.float16, device_type="cuda"):
                noisy_wav = noisy_wav.to(self.rank)
                rev_wav = rev_wav.to(self.rank)
                dp_wav = dp_wav.to(self.rank)
                angle_class = angle_class.to(self.rank)
                radius_class = radius_class.to(self.rank)
                room_params = room_params.to(self.rank) if room_params is not None else None

                input_complex = self.transformfunc.stft(
                    noisy_wav, output_type="complex"
                ).to(dtype=torch.complex64)
                target_complex = self.transformfunc.stft(
                    dp_wav, output_type="complex"
                ).to(dtype=torch.complex64)
                reverb_complex = self.transformfunc.stft(
                    rev_wav, output_type="complex"
                ).to(dtype=torch.complex64)

                input_ft = self.transformfunc.preprocess(input_complex)

                (
                    est_spch_ft,
                    est_ctf_ft,
                    est_reverb_ft,
                    angle_logits,
                    radius_logits,
                ) = self.compiled_model(input_ft, room_params=room_params)

                est_spch = self.transformfunc.postprocess(est_spch_ft).to(
                    dtype=torch.complex64
                )
                est_ctf = self.transformfunc.postprocess(est_ctf_ft).to(
                    dtype=torch.complex64
                )
                est_reverb = self.transformfunc.postprocess(est_reverb_ft).to(
                    dtype=torch.complex64
                )
                L = est_ctf.shape[-1]
                recon = torchaudio.functional.convolve(
                    target_complex, est_ctf, mode="full"
                )[..., : target_complex.shape[-1]]

                ######## calculate losses ########
                loss_cln = self.lossF(est_spch, target_complex)
                loss_rvb = self.lossF(est_reverb, reverb_complex)
                loss_rec = self.lossF(recon, reverb_complex)
                loss_angle = torch.tensor(0.0, device=self.rank)
                loss_radius = torch.tensor(0.0, device=self.rank)
                if self.loss_w_angle > 0:
                    valid_angle = angle_class >= 0
                    if valid_angle.any():
                        loss_angle = self.angle_loss(
                            angle_logits[valid_angle], angle_class[valid_angle]
                        )
                if self.loss_w_radius > 0:
                    valid_radius = radius_class >= 0
                    if valid_radius.any():
                        loss_radius = self.radius_loss(
                            radius_logits[valid_radius], radius_class[valid_radius]
                        )

                loss_gd = (
                    self.loss_w_cln * loss_cln
                    + self.loss_w_rec * loss_rec
                    + self.loss_w_rvb * loss_rvb
                    + self.loss_w_angle * loss_angle
                    + self.loss_w_radius * loss_radius
                )

            self.scaler.scale(loss_gd).backward(retain_graph=True)

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_grad_norm_value
            )

            self.scaler.step(self.optimizer)

            self.scaler.update()
            self.steps += 1

            if self.scheduler:
                self.scheduler.step(self.steps)

            loss_total = loss_total + loss_gd.item()

            if (index) % 100 == 0:

                if self.rank == 0:
                    self._log_metrics({
                        "Train_step/Lr": self.optimizer.param_groups[0]["lr"],
                        "Train_step/Loss": loss_gd.item(),
                        "Train_step/loss_cln": loss_cln.item(),
                        "Train_step/loss_rvb": loss_rvb.item(),
                        "Train_step/loss_rec": loss_rec.item(),
                        "Train_step/loss_angle": loss_angle.item(),
                        "Train_step/loss_radius": loss_radius.item(),
                    }, step=self.steps)

        if self.rank == 0:
            self._log_metrics({
                "Train_epoch/Loss": loss_total / len(self.train_dataloader),
            }, step=self.steps)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        loss_total = 0.0
        loss_total_cln = 0.0
        loss_total_rvb = 0.0
        loss_total_rec = 0.0
        loss_total_angle = 0.0
        loss_total_radius = 0.0
        # Running sums (over batches, reduced across DDP ranks) of the per-batch
        # mean target entropy H(soft_target). Used to convert soft-label CE back
        # into KL divergence for cross-run comparability: KL has a fixed zero
        # floor regardless of the smoothing sigma. For hard-label CE the soft
        # target is one-hot, so H == 0 and KL == CE.
        entropy_total_angle = 0.0
        entropy_total_radius = 0.0
        error_total_angle = 0.0
        error_total_radius = 0.0
        num_valid_angle_samples = 0
        num_valid_radius_samples = 0

        for index, (noisy_wav, rev_wav, dp_wav, fpath, rir_path, angle_class, radius_class, room_params) in (
            enumerate(tqdm(self.valid_dataloader, desc="Validating"))
            if self.rank == 0
            else enumerate(self.valid_dataloader)
        ):
            noisy_wav = noisy_wav.to(self.rank)
            rev_wav = rev_wav.to(self.rank)
            dp_wav = dp_wav.to(self.rank)
            angle_class = angle_class.to(self.rank)
            radius_class = radius_class.to(self.rank)
            room_params = room_params.to(self.rank) if room_params is not None else None

            input_complex = self.transformfunc.stft(noisy_wav, output_type="complex")
            target_complex = self.transformfunc.stft(dp_wav, output_type="complex")
            reverb_complex = self.transformfunc.stft(rev_wav, output_type="complex")

            input_ft = self.transformfunc.preprocess(input_complex)
            (
                est_spch_ft,
                est_ctf_ft,
                est_reverb_ft,
                angle_logits,
                radius_logits,
            ) = self.model(input_ft, room_params=room_params)

            est_spch = self.transformfunc.postprocess(est_spch_ft)
            est_ctf = self.transformfunc.postprocess(est_ctf_ft)
            est_reverb = self.transformfunc.postprocess(est_reverb_ft)

            L = est_ctf.shape[-1]
            recon = torchaudio.functional.convolve(
                target_complex, est_ctf, mode="full"
            )[..., : target_complex.shape[-1]]

            ######## calculate losses ########
            loss_cln = self.lossF(est_spch, target_complex)
            loss_rvb = self.lossF(est_reverb, reverb_complex)
            loss_rec = self.lossF(recon, reverb_complex)
            loss_angle = torch.tensor(0.0, device=self.rank)
            loss_radius = torch.tensor(0.0, device=self.rank)

            # Per-batch mean H(soft_target). Only meaningful when the corresponding
            # head uses GaussianLabelSmoothingLoss; otherwise this stays at 0,
            # which is the correct H value for a one-hot (hard-CE) target.
            mean_entropy_angle_batch = torch.tensor(0.0, device=self.rank)
            mean_entropy_radius_batch = torch.tensor(0.0, device=self.rank)

            angle_error_batch = torch.tensor(0.0, device=self.rank)
            radius_error_batch = torch.tensor(0.0, device=self.rank)
            angle_count_batch = torch.tensor(0, device=self.rank, dtype=torch.int64)
            radius_count_batch = torch.tensor(0, device=self.rank, dtype=torch.int64)

            if self.loss_w_angle > 0:
                valid_angle = angle_class >= 0
                if valid_angle.any():
                    valid_angle_labels = angle_class[valid_angle]
                    loss_angle = self.angle_loss(
                        angle_logits[valid_angle], valid_angle_labels
                    )
                    if isinstance(self.angle_loss, GaussianLabelSmoothingLoss):
                        mean_entropy_angle_batch = self.angle_loss.target_entropy(
                            valid_angle_labels
                        ).mean()
                    angle_predictions = torch.argmax(angle_logits[valid_angle], dim=1)
                    angle_error_batch = torch.abs(
                        angle_predictions - valid_angle_labels
                    ).float().sum()
                    angle_count_batch = valid_angle.sum()

            if self.loss_w_radius > 0:
                valid_radius = radius_class >= 0
                if valid_radius.any():
                    valid_radius_labels = radius_class[valid_radius]
                    loss_radius = self.radius_loss(
                        radius_logits[valid_radius], valid_radius_labels
                    )
                    if isinstance(self.radius_loss, GaussianLabelSmoothingLoss):
                        mean_entropy_radius_batch = self.radius_loss.target_entropy(
                            valid_radius_labels
                        ).mean()
                    radius_predictions = torch.argmax(radius_logits[valid_radius], dim=1)
                    radius_error_batch = torch.abs(
                        radius_predictions - valid_radius_labels
                    ).float().sum()
                    radius_count_batch = valid_radius.sum()

            loss = (
                self.loss_w_cln * loss_cln
                + self.loss_w_rec * loss_rec
                + self.loss_w_rvb * loss_rvb
                + self.loss_w_angle * loss_angle
                + self.loss_w_radius * loss_radius
            )

            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_cln, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_rec, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_rvb, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_angle, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_radius, op=dist.ReduceOp.SUM)
            dist.all_reduce(mean_entropy_angle_batch, op=dist.ReduceOp.SUM)
            dist.all_reduce(mean_entropy_radius_batch, op=dist.ReduceOp.SUM)
            dist.all_reduce(angle_error_batch, op=dist.ReduceOp.SUM)
            dist.all_reduce(radius_error_batch, op=dist.ReduceOp.SUM)
            dist.all_reduce(angle_count_batch, op=dist.ReduceOp.SUM)
            dist.all_reduce(radius_count_batch, op=dist.ReduceOp.SUM)

            loss_total += loss
            loss_total_cln += loss_cln
            loss_total_rvb += loss_rvb
            loss_total_rec += loss_rec
            loss_total_angle += loss_angle
            loss_total_radius += loss_radius
            # Entropy sums are accumulated as sums-across-ranks of per-batch means.
            # Denominator of num_batches * world_size matches the CE averaging so
            # that avg_loss_*_ce - avg_entropy_* is a well-defined KL >= 0.
            entropy_total_angle += mean_entropy_angle_batch
            entropy_total_radius += mean_entropy_radius_batch
            error_total_angle += angle_error_batch
            error_total_radius += radius_error_batch
            num_valid_angle_samples += angle_count_batch.item()
            num_valid_radius_samples += radius_count_batch.item()
            if self.rank == 0:
                log_dict = {
                    "Valid_epoch/Loss": loss_total / len(self.valid_dataloader) / dist.get_world_size(),
                    "Valid_epoch/loss_cln": loss_total_cln / len(self.valid_dataloader) / dist.get_world_size(),
                    "Valid_epoch/loss_rvb": loss_total_rvb / len(self.valid_dataloader) / dist.get_world_size(),
                    "Valid_epoch/loss_rec": loss_total_rec / len(self.valid_dataloader) / dist.get_world_size(),
                    "Valid_epoch/loss_angle": loss_total_angle / len(self.valid_dataloader) / dist.get_world_size(),
                    "Valid_epoch/loss_radius": loss_total_radius / len(self.valid_dataloader) / dist.get_world_size(),
                }

                if num_valid_angle_samples > 0:
                    log_dict["Valid_epoch/error_angle_abs_deg"] = (
                        error_total_angle / num_valid_angle_samples
                    ).item()
                if num_valid_radius_samples > 0:
                    # Convert radius class-index error to meters via rad_resolution.
                    log_dict["Valid_epoch/error_radius_abs_m"] = (
                        error_total_radius / num_valid_radius_samples
                    ).item() * self.rad_resolution

                self._log_metrics(log_dict, step=self.steps)

                if index == 0:
                    self._log_metrics({
                        "Valid_epoch/target_cln": wandb.Image(plot_spectrogram(target_complex[0].abs().squeeze())),
                        "Valid_epoch/input": wandb.Image(plot_spectrogram(input_complex[0].abs().squeeze())),
                        "Valid_epoch/target_reverb": wandb.Image(plot_spectrogram(reverb_complex[0].abs().squeeze())),
                        "Valid_epoch/est_cln": wandb.Image(plot_spectrogram(est_spch[0].abs().squeeze())),
                        "Valid_epoch/est_reverb": wandb.Image(plot_spectrogram(recon[0].abs().squeeze())),
                        "Valid_epoch/est_ctf": wandb.Image(plot_spectrogram(est_ctf[0].abs().squeeze())),
                    }, step=self.steps)

        num_batches = len(self.valid_dataloader)
        world_size = dist.get_world_size()
        avg_loss_total = (loss_total / num_batches / world_size).item()
        avg_loss_angle_ce = (loss_total_angle / num_batches / world_size).item()
        avg_loss_radius_ce = (loss_total_radius / num_batches / world_size).item()
        avg_loss_localization_ce = avg_loss_angle_ce + avg_loss_radius_ce

        # Mean H(soft_target) over the epoch, per head. Use the SAME denominator
        # as the CE averaging (num_batches * world_size) so that KL = CE - H is
        # well-defined and non-negative: if a rank/batch had no valid labels,
        # both loss_angle and mean_entropy_angle_batch are 0 and cancel out.
        avg_entropy_angle = (entropy_total_angle / num_batches / world_size).item()
        avg_entropy_radius = (entropy_total_radius / num_batches / world_size).item()

        # KL divergence form of the classification losses: CE - H(target). This
        # zeros out at a perfect predictor regardless of the label-smoothing sigma,
        # so it's cross-run comparable. For hard-label CE these equal the raw CE
        # (since H(one-hot) = 0).
        avg_loss_angle_kl = max(avg_loss_angle_ce - avg_entropy_angle, 0.0)
        avg_loss_radius_kl = max(avg_loss_radius_ce - avg_entropy_radius, 0.0)
        avg_loss_localization_kl = avg_loss_angle_kl + avg_loss_radius_kl

        # Absolute errors. If a split has zero valid labels, fall back to +inf so the
        # metric never silently treats "no data" as "perfect".
        if num_valid_angle_samples > 0:
            avg_angle_error_deg = (
                error_total_angle / num_valid_angle_samples
            ).item()
        else:
            avg_angle_error_deg = float("inf")

        if num_valid_radius_samples > 0:
            avg_radius_error_m = (
                error_total_radius / num_valid_radius_samples
            ).item() * self.rad_resolution
        else:
            avg_radius_error_m = float("inf")

        avg_localization_error = (
            avg_angle_error_deg + self.best_metric_radius_weight * avg_radius_error_m
        )

        # NOTE: angle_ce / radius_ce / localization_ce resolve to the KL form so
        # the "best" scalar has a sigma-independent zero floor. Under hard-label CE
        # KL == CE, so runs without label smoothing are unaffected.
        metric_map = {
            "total": avg_loss_total,
            "angle_ce": avg_loss_angle_kl,
            "radius_ce": avg_loss_radius_kl,
            "localization_ce": avg_loss_localization_kl,
            "angle_error": avg_angle_error_deg,
            "radius_error": avg_radius_error_m,
            "localization": avg_localization_error,
        }
        best_metric_value = metric_map[self.best_metric_name]

        # Log the full metric panel so the "best" curve, raw CE, KL form and
        # argmax errors are all visible in WandB regardless of which option the
        # user selected for checkpoint selection.
        if self.rank == 0:
            self._log_metrics(
                {
                    "Valid_epoch/best_metric": best_metric_value,
                    "Valid_epoch/error_localization": avg_localization_error,
                    "Valid_epoch/loss_localization_ce": avg_loss_localization_ce,
                    "Valid_epoch/loss_angle_kl": avg_loss_angle_kl,
                    "Valid_epoch/loss_radius_kl": avg_loss_radius_kl,
                    "Valid_epoch/loss_localization_kl": avg_loss_localization_kl,
                    "Valid_epoch/target_entropy_angle": avg_entropy_angle,
                    "Valid_epoch/target_entropy_radius": avg_entropy_radius,
                },
                step=self.steps,
            )

        return best_metric_value
