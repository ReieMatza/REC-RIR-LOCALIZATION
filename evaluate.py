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

import warnings
warnings.filterwarnings("ignore")


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
    
    # Evaluation loop
    print(f"\nEvaluating on {num_samples} samples...")
    loss_total = 0.0
    loss_total_cln = 0.0
    loss_total_rvb = 0.0
    loss_total_rec = 0.0
    
    samples_evaluated = 0
    
    with torch.no_grad():
        for index, (noisy_wav, rev_wav, dp_wav, fpath) in enumerate(tqdm(valid_dataloader, desc="Evaluating")):
            if samples_evaluated >= num_samples:
                break
            
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
            est_spch_ft, est_ctf_ft, est_reverb_ft = model(input_ft)
            
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
            
            # Accumulate losses
            loss_total += loss.item()
            loss_total_cln += loss_cln.item()
            loss_total_rvb += loss_rvb.item()
            loss_total_rec += loss_rec.item()
            
            samples_evaluated += noisy_wav.shape[0]  # batch size
    
    # Calculate average losses
    avg_loss_total = loss_total / samples_evaluated
    avg_loss_cln = loss_total_cln / samples_evaluated
    avg_loss_rvb = loss_total_rvb / samples_evaluated
    avg_loss_rec = loss_total_rec / samples_evaluated
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Evaluation Results ({samples_evaluated} samples):")
    print(f"{'='*50}")
    print(f"Total Loss:     {avg_loss_total:.6f}")
    print(f"Clean Loss:     {avg_loss_cln:.6f}")
    print(f"Reverb Loss:    {avg_loss_rvb:.6f}")
    print(f"Recon Loss:     {avg_loss_rec:.6f}")
    print(f"{'='*50}\n")


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
        required=True,
        type=str,
        help="Path to model checkpoint (required)"
    )
    parser.add_argument(
        "-n", "--num_samples",
        type=int,
        default=20,
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
