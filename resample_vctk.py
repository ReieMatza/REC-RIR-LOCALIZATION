#!/usr/bin/env python3
"""
Resample VCTK corpus, EARS dataset, and noise audio files from various sample rates to 16kHz.
This script processes all .wav files in the VCTK dataset, EARS dataset, and noise datasets and resamples them in-place.
"""

import os
import sys
from pathlib import Path
import torchaudio
from tqdm import tqdm
import argparse


def resample_file(input_path: Path, target_sr: int = 16000):
    """
    Resample a single audio file to target sample rate.
    
    Args:
        input_path: Path to the input audio file
        target_sr: Target sample rate (default: 16000)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load audio file using soundfile backend
        waveform, sample_rate = torchaudio.load(str(input_path), backend="soundfile")
        
        # Check if resampling is needed
        if sample_rate == target_sr:
            return True
        
        # Resample if needed
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        resampled_waveform = resampler(waveform)
        
        # Save the resampled audio (overwrite original)
        torchaudio.save(
            str(input_path),
            resampled_waveform,
            target_sr,
            backend="soundfile"
        )
        
        return True
    except Exception as e:
        print(f"\nError processing {input_path}: {e}", file=sys.stderr)
        return False


def resample_directory(root_dir: Path, target_sr: int = 16000, dry_run: bool = False):
    """
    Resample all .wav files in a directory tree.
    
    Args:
        root_dir: Root directory containing audio files
        target_sr: Target sample rate (default: 16000)
        dry_run: If True, only print what would be done without actually resampling
    """
    # Find all .wav files
    wav_files = list(root_dir.rglob("*.wav"))
    
    if not wav_files:
        print(f"No .wav files found in {root_dir}")
        return
    
    print(f"Found {len(wav_files)} .wav files")
    
    if dry_run:
        print("DRY RUN MODE: No files will be modified")
        # Check sample rates
        sample_rates = {}
        for wav_file in tqdm(wav_files[:100], desc="Checking sample rates (first 100)"):
            try:
                info = torchaudio.info(str(wav_file), backend="soundfile")
                sr = info.sample_rate
                sample_rates[sr] = sample_rates.get(sr, 0) + 1
            except Exception as e:
                print(f"\nError checking {wav_file}: {e}")
        
        print(f"\nSample rate distribution (first 100 files):")
        for sr, count in sorted(sample_rates.items()):
            print(f"  {sr} Hz: {count} files")
        return
    
    # Process files
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for wav_file in tqdm(wav_files, desc="Resampling audio files"):
        try:
            # Check current sample rate
            info = torchaudio.info(str(wav_file), backend="soundfile")
            current_sr = info.sample_rate
            
            if current_sr == target_sr:
                skip_count += 1
                continue
            
            # Resample
            if resample_file(wav_file, target_sr):
                success_count += 1
            else:
                error_count += 1
        except Exception as e:
            print(f"\nError processing {wav_file}: {e}", file=sys.stderr)
            error_count += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Resampling complete!")
    print(f"  Successfully resampled: {success_count} files")
    print(f"  Already at {target_sr} Hz (skipped): {skip_count} files")
    print(f"  Errors: {error_count} files")
    print(f"  Total processed: {len(wav_files)} files")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Resample VCTK corpus, EARS dataset, and noise audio files to 16kHz"
    )
    parser.add_argument(
        "--vctk-dir",
        type=str,
        default="/storage/reie/data/rec-rir/speech/VCTK/VCTK-Corpus/VCTK-Corpus/wav48",
        help="Directory containing the VCTK speech audio files to resample"
    )
    parser.add_argument(
        "--ears-dir",
        type=str,
        default="/storage/reie/data/rec-rir/speech/EARS",
        help="Directory containing the EARS dataset audio files to resample"
    )
    parser.add_argument(
        "--noise-dir",
        type=str,
        default="/storage/reie/data/rec-rir/noise/NoiseX-92",
        help="Directory containing the noise audio files to resample"
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=16000,
        help="Target sample rate (default: 16000)"
    )
    parser.add_argument(
        "--skip-vctk",
        action="store_true",
        help="Skip processing VCTK speech files"
    )
    parser.add_argument(
        "--skip-ears",
        action="store_true",
        help="Skip processing EARS dataset files"
    )
    parser.add_argument(
        "--skip-noise",
        action="store_true",
        help="Skip processing noise files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode: check sample rates without modifying files"
    )
    
    args = parser.parse_args()
    
    print(f"Target sample rate: {args.target_sr} Hz")
    print(f"Dry run: {args.dry_run}")
    print()
    
    # Process VCTK files unless skipped
    if not args.skip_vctk:
        vctk_dir = Path(args.vctk_dir)
        
        if not vctk_dir.exists():
            print(f"Warning: VCTK directory {vctk_dir} does not exist! Skipping VCTK processing.", file=sys.stderr)
        elif not vctk_dir.is_dir():
            print(f"Warning: {vctk_dir} is not a directory! Skipping VCTK processing.", file=sys.stderr)
        else:
            print(f"Processing VCTK directory: {vctk_dir}")
            resample_directory(vctk_dir, args.target_sr, args.dry_run)
            print()
    
    # Process EARS files unless skipped
    if not args.skip_ears:
        ears_dir = Path(args.ears_dir)
        
        if not ears_dir.exists():
            print(f"Warning: EARS directory {ears_dir} does not exist! Skipping EARS processing.", file=sys.stderr)
        elif not ears_dir.is_dir():
            print(f"Warning: {ears_dir} is not a directory! Skipping EARS processing.", file=sys.stderr)
        else:
            print(f"Processing EARS directory: {ears_dir}")
            resample_directory(ears_dir, args.target_sr, args.dry_run)
            print()
    
    # Process noise files unless skipped
    if not args.skip_noise:
        noise_dir = Path(args.noise_dir)
        
        if not noise_dir.exists():
            print(f"Warning: Noise directory {noise_dir} does not exist! Skipping noise processing.", file=sys.stderr)
        elif not noise_dir.is_dir():
            print(f"Warning: {noise_dir} is not a directory! Skipping noise processing.", file=sys.stderr)
        else:
            print(f"Processing noise directory: {noise_dir}")
            resample_directory(noise_dir, args.target_sr, args.dry_run)


if __name__ == "__main__":
    main()
