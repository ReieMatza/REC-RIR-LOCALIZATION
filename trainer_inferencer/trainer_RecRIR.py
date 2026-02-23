import torch
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
from .base_trainer_RecRIR import BaseTrainer
from .utils import plot_spectrogram
import torchaudio
from model.lossF import RIMag_loss, MSE_loss_complex
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

        self.loss_w_rvb = config["trainer"]["train"]["loss_w_rvb"]
        self.loss_w_cln = config["trainer"]["train"]["loss_w_cln"]
        self.loss_w_rec = config["trainer"]["train"]["loss_w_rec"]
        self.loss_w_angle = config["trainer"]["train"].get("loss_w_angle", 0.0)
        self.loss_w_radius = config["trainer"]["train"].get("loss_w_radius", 0.0)
        self.classification_loss = nn.CrossEntropyLoss()

        self.scaler = GradScaler()
        if config["trainer"]["train"]["loss_type"] == "RIMag":
            self.lossF = lossF_RIMag
        elif config["trainer"]["train"]["loss_type"] == "MSE":
            self.lossF = lossF_MSE

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
                        loss_angle = self.classification_loss(
                            angle_logits[valid_angle], angle_class[valid_angle]
                        )
                if self.loss_w_radius > 0:
                    valid_radius = radius_class >= 0
                    if valid_radius.any():
                        loss_radius = self.classification_loss(
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
                    wandb.log({
                        "Train_step/Lr": self.optimizer.param_groups[0]["lr"],
                        "Train_step/Loss": loss_gd.item(),
                        "Train_step/loss_cln": loss_cln.item(),
                        "Train_step/loss_rvb": loss_rvb.item(),
                        "Train_step/loss_rec": loss_rec.item(),
                        "Train_step/loss_angle": loss_angle.item(),
                        "Train_step/loss_radius": loss_radius.item(),
                    }, step=self.steps)

        if self.rank == 0:
            wandb.log({
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
            
            # Calculate absolute errors
            angle_error_batch = torch.tensor(0.0, device=self.rank)
            radius_error_batch = torch.tensor(0.0, device=self.rank)
            angle_count_batch = torch.tensor(0, device=self.rank, dtype=torch.int64)
            radius_count_batch = torch.tensor(0, device=self.rank, dtype=torch.int64)
            
            if self.loss_w_angle > 0:
                valid_angle = angle_class >= 0
                if valid_angle.any():
                    loss_angle = self.classification_loss(
                        angle_logits[valid_angle], angle_class[valid_angle]
                    )
                    # Calculate absolute angle error
                    angle_predictions = torch.argmax(angle_logits[valid_angle], dim=1)
                    angle_error_batch = torch.abs(angle_predictions - angle_class[valid_angle]).float().sum()
                    angle_count_batch = valid_angle.sum()
                    
            if self.loss_w_radius > 0:
                valid_radius = radius_class >= 0
                if valid_radius.any():
                    loss_radius = self.classification_loss(
                        radius_logits[valid_radius], radius_class[valid_radius]
                    )
                    # Calculate absolute radius error
                    radius_predictions = torch.argmax(radius_logits[valid_radius], dim=1)
                    radius_error_batch = torch.abs(radius_predictions - radius_class[valid_radius]).float().sum()
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
                
                # Add absolute error metrics if we have valid samples
                if num_valid_angle_samples > 0:
                    log_dict["Valid_epoch/error_angle_abs_deg"] = (
                        error_total_angle / num_valid_angle_samples
                    ).item()
                if num_valid_radius_samples > 0:
                    log_dict["Valid_epoch/error_radius_abs_m"] = (
                        error_total_radius / num_valid_radius_samples
                    ).item()
                
                wandb.log(log_dict, step=self.steps)

                if index == 0:
                    wandb.log({
                        "Valid_epoch/target_cln": wandb.Image(plot_spectrogram(target_complex[0].abs().squeeze())),
                        "Valid_epoch/input": wandb.Image(plot_spectrogram(input_complex[0].abs().squeeze())),
                        "Valid_epoch/target_reverb": wandb.Image(plot_spectrogram(reverb_complex[0].abs().squeeze())),
                        "Valid_epoch/est_cln": wandb.Image(plot_spectrogram(est_spch[0].abs().squeeze())),
                        "Valid_epoch/est_reverb": wandb.Image(plot_spectrogram(recon[0].abs().squeeze())),
                        "Valid_epoch/est_ctf": wandb.Image(plot_spectrogram(est_ctf[0].abs().squeeze())),
                    }, step=self.steps)

        return loss_total / len(self.valid_dataloader) / dist.get_world_size()
