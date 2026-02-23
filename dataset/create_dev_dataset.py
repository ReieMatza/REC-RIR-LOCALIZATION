#!/usr/bin/env python3
"""
Script to create smaller development datasets (5% of original size) from the full dataset files.
This is useful for quick development and testing.
"""

import os
import random
import argparse
from pathlib import Path


def sample_file_lines(input_file, output_file, sample_ratio=0.05, seed=42):
    """
    Sample a percentage of lines from an input file and write to output file.
    
    Args:
        input_file: Path to input file
        output_file: Path to output file
        sample_ratio: Fraction of lines to sample (default 0.05 = 5%)
        seed: Random seed for reproducibility
    """
    # Read all non-empty lines
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if not lines:
        print(f"Warning: {input_file} is empty, skipping...")
        return 0
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Calculate number of lines to sample
    num_lines = len(lines)
    num_sample = max(1, int(num_lines * sample_ratio))  # At least 1 line
    
    # Randomly sample lines
    sampled_lines = random.sample(lines, num_sample)
    
    # Write sampled lines to output file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in sampled_lines:
            f.write(line + '\n')
    
    print(f"Sampled {num_sample}/{num_lines} lines ({num_sample/num_lines*100:.2f}%) from {input_file}")
    print(f"  -> {output_file}")
    
    return num_sample


def main():
    parser = argparse.ArgumentParser(
        description='Create smaller development datasets (5% of original size)'
    )
    parser.add_argument(
        '--config-dir',
        type=str,
        default='config',
        help='Directory containing the dataset text files (default: config)'
    )
    parser.add_argument(
        '--sample-ratio',
        type=float,
        default=0.25,
        help='Fraction of lines to sample (default: 0.05 = 5%%)'
    )
    parser.add_argument(
        '--suffix',
        type=str,
        default='_quick',
        help='Suffix to add to output filenames (default: _dev)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Get the project root directory (parent of dataset directory)
    project_root = Path(__file__).parent.parent
    config_dir = project_root / args.config_dir
    
    # List of dataset files to process
    dataset_files = [
        'reie_train_rir.txt',
        'reie_validation_rir.txt',
        'reie_train_speech.txt',
        'reie_validation_speech.txt',
        'reie_train_noise.txt',
        'reie_validation_noise.txt',
    ]
    
    print(f"Creating development datasets ({args.sample_ratio*100:.1f}% of original size)...")
    print(f"Config directory: {config_dir}")
    print(f"Output suffix: {args.suffix}")
    print(f"Random seed: {args.seed}")
    print("-" * 60)
    
    total_original = 0
    total_sampled = 0
    
    for filename in dataset_files:
        input_file = config_dir / filename
        
        if not input_file.exists():
            print(f"Warning: {input_file} does not exist, skipping...")
            continue
        
        # Create output filename
        name_parts = filename.rsplit('.', 1)
        if len(name_parts) == 2:
            output_filename = f"{name_parts[0]}{args.suffix}.{name_parts[1]}"
        else:
            output_filename = f"{filename}{args.suffix}"
        
        output_file = config_dir / output_filename
        
        # Count original lines
        with open(input_file, 'r', encoding='utf-8') as f:
            original_count = sum(1 for line in f if line.strip())
        total_original += original_count
        
        # Sample and write
        sampled_count = sample_file_lines(input_file, output_file, args.sample_ratio, args.seed)
        total_sampled += sampled_count
    
    print("-" * 60)
    print(f"Summary:")
    print(f"  Original total lines: {total_original:,}")
    print(f"  Sampled total lines: {total_sampled:,}")
    print(f"  Reduction: {(1 - total_sampled/total_original)*100:.2f}%")
    print(f"\nDevelopment dataset files created with suffix '{args.suffix}'")


if __name__ == '__main__':
    main()
