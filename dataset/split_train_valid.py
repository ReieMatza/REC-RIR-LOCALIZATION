#!/usr/bin/env python3
# @author: Auto-generated
# @description: Script to split a file list into train and validation sets

import random
from pathlib import Path
from jsonargparse import ArgumentParser


def split_file_list(input_file: str, train_file: str, valid_file: str, 
                   train_ratio: float = 0.9, seed: int = 42, 
                   shuffle: bool = True, preserve_speaker: bool = False):
    """
    Split a file list into train and validation sets.
    
    Args:
        input_file: Path to input file with file paths (one per line)
        train_file: Path to output train file
        valid_file: Path to output validation file
        train_ratio: Ratio of data for training (default: 0.9 for 90/10 split)
        seed: Random seed for reproducibility (default: 42)
        shuffle: Whether to shuffle the data before splitting
        preserve_speaker: If True, group files by speaker and split by speaker
                         (useful for speaker-dependent datasets)
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"✗ Error: Input file does not exist: {input_file}")
        return False
    
    print(f"Reading file: {input_file}")
    
    # Read all lines
    with open(input_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    total_lines = len(lines)
    print(f"Found {total_lines} file paths")
    
    if total_lines == 0:
        print("✗ Error: Input file is empty")
        return False
    
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
    
    if preserve_speaker:
        # Group files by speaker (assuming path contains speaker identifier)
        # For EARS: /path/to/EARS/p001/file.wav -> speaker is p001
        # For VCTK: /path/to/VCTK/p225/file.wav -> speaker is p225
        speaker_groups = {}
        for line in lines:
            # Try to extract speaker ID from path
            parts = Path(line).parts
            speaker_id = None
            for part in parts:
                if part.startswith('p') and len(part) >= 2 and part[1:].isdigit():
                    speaker_id = part
                    break
            
            if speaker_id is None:
                # Fallback: use parent directory name
                speaker_id = Path(line).parent.name
            
            if speaker_id not in speaker_groups:
                speaker_groups[speaker_id] = []
            speaker_groups[speaker_id].append(line)
        
        print(f"Grouped into {len(speaker_groups)} speakers")
        
        # Shuffle speaker groups
        if shuffle:
            speaker_list = list(speaker_groups.items())
            random.shuffle(speaker_list)
        else:
            speaker_list = list(speaker_groups.items())
        
        # Split speakers (not individual files)
        num_speakers = len(speaker_list)
        train_speaker_count = int(num_speakers * train_ratio)
        
        train_speakers = speaker_list[:train_speaker_count]
        valid_speakers = speaker_list[train_speaker_count:]
        
        # Collect files from train and validation speakers
        train_lines = []
        valid_lines = []
        
        for speaker_id, files in train_speakers:
            train_lines.extend(files)
        
        for speaker_id, files in valid_speakers:
            valid_lines.extend(files)
        
        # Shuffle within each set
        if shuffle:
            random.shuffle(train_lines)
            random.shuffle(valid_lines)
        
        print(f"Train: {len(train_lines)} files from {len(train_speakers)} speakers")
        print(f"Validation: {len(valid_lines)} files from {len(valid_speakers)} speakers")
        
    else:
        # Simple random split
        if shuffle:
            lines_shuffled = lines.copy()
            random.shuffle(lines_shuffled)
        else:
            lines_shuffled = lines
        
        # Calculate split point
        train_count = int(total_lines * train_ratio)
        
        train_lines = lines_shuffled[:train_count]
        valid_lines = lines_shuffled[train_count:]
        
        print(f"Train: {len(train_lines)} files ({len(train_lines)/total_lines*100:.1f}%)")
        print(f"Validation: {len(valid_lines)} files ({len(valid_lines)/total_lines*100:.1f}%)")
    
    # Write train file
    train_path = Path(train_file)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nWriting train file: {train_file}")
    with open(train_path, 'w') as f:
        for line in train_lines:
            f.write(line + '\n')
    
    # Write validation file
    valid_path = Path(valid_file)
    valid_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Writing validation file: {valid_file}")
    with open(valid_path, 'w') as f:
        for line in valid_lines:
            f.write(line + '\n')
    
    print("\n✓ Split completed successfully!")
    print(f"  Train: {len(train_lines)} files")
    print(f"  Validation: {len(valid_lines)} files")
    print(f"  Ratio: {len(train_lines)/total_lines*100:.1f}% / {len(valid_lines)/total_lines*100:.1f}%")
    
    return True


def main(input_file: str, train_file: str, valid_file: str, 
         train_ratio: float = 0.9, seed: int = 42, 
         shuffle: bool = True, preserve_speaker: bool = False):
    """
    Split a file list into train and validation sets.
    
    Args:
        input_file: Path to input file with file paths (one per line)
        train_file: Path to output train file
        valid_file: Path to output validation file
        train_ratio: Ratio of data for training (default: 0.9 for 90/10 split)
        seed: Random seed for reproducibility (default: 42)
        shuffle: Whether to shuffle the data before splitting
        preserve_speaker: If True, group files by speaker and split by speaker
    """
    if train_ratio <= 0 or train_ratio >= 1:
        print("✗ Error: train_ratio must be between 0 and 1")
        return False
    
    return split_file_list(
        input_file=input_file,
        train_file=train_file,
        valid_file=valid_file,
        train_ratio=train_ratio,
        seed=seed,
        shuffle=shuffle,
        preserve_speaker=preserve_speaker
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Split a file list into train and validation sets")
    parser.add_argument(
        "-i", "--input_file",
        required=True,
        type=str,
        help="Input file with file paths (one per line)"
    )
    parser.add_argument(
        "-t", "--train_file",
        required=True,
        type=str,
        help="Output file for training set"
    )
    parser.add_argument(
        "-v", "--valid_file",
        required=True,
        type=str,
        help="Output file for validation set"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Ratio of data for training (default: 0.9 for 90/10 split)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_false",
        dest="shuffle",
        help="Do not shuffle before splitting"
    )
    parser.add_argument(
        "--preserve-speaker",
        action="store_true",
        help="Group files by speaker and split by speaker (keeps all files from same speaker together)"
    )
    
    args = parser.parse_args()
    main(**args)
    
    """
    Usage examples:
    
    # Basic 90/10 split
    python /mnt/c/Users/reiem/PythonProjects/Rec-RIR/dataset/split_train_valid.py -i config/reie_train_spch.txt -t config/train_spch.txt -v config/valid_spch.txt
    
    # Custom split ratio (80/20)
    python split_train_valid.py -i config/reie_train_spch.txt -t config/train_spch.txt -v config/valid_spch.txt --train_ratio 0.8
    
    # Preserve speaker groups (all files from same speaker stay together)
    python split_train_valid.py -i config/reie_train_spch.txt -t config/train_spch.txt -v config/valid_spch.txt --preserve-speaker
    
    # Use different random seed
    python split_train_valid.py -i config/reie_train_spch.txt -t config/train_spch.txt -v config/valid_spch.txt --seed 123
    
    # No shuffling (sequential split)
    python split_train_valid.py -i config/reie_train_spch.txt -t config/train_spch.txt -v config/valid_spch.txt --no-shuffle
    """

