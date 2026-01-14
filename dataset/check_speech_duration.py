#!/usr/bin/env python3
# @author: Auto-generated
# @description: Script to calculate total duration of all WAV files in a directory

import sys
from pathlib import Path
from jsonargparse import ArgumentParser
from glob import glob
import librosa
from tqdm import tqdm


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    parts = []
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if secs > 0 or len(parts) == 0:
        parts.append(f"{secs:.2f} second{'s' if secs != 1 else ''}")
    
    return ", ".join(parts)


def get_audio_duration(filepath: Path) -> float:
    """Get duration of an audio file in seconds."""
    try:
        # Use librosa to get duration (doesn't load full audio into memory)
        duration = librosa.get_duration(path=str(filepath))
        return duration
    except Exception as e:
        print(f"\n⚠ Warning: Could not read {filepath}: {e}")
        return 0.0


def calculate_total_duration(directory: Path, recursive: bool = True, 
                            extensions: list = None, show_progress: bool = True) -> dict:
    """
    Calculate total duration of all audio files in a directory.
    
    Args:
        directory: Directory to search for audio files
        recursive: If True, search subdirectories recursively
        extensions: List of file extensions to include (default: ['.wav', '.flac', '.mp3'])
        show_progress: If True, show progress bar
    
    Returns:
        Dictionary with statistics about the audio files
    """
    if extensions is None:
        extensions = ['.wav', '.flac', '.mp3']
    
    # Normalize extensions (add dot if missing, lowercase)
    extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                  for ext in extensions]
    
    print(f"\nSearching for audio files with extensions: {', '.join(extensions)}")
    print(f"Directory: {directory}")
    print(f"Recursive: {recursive}")
    
    # Find all audio files
    audio_files = []
    for ext in extensions:
        if recursive:
            pattern = f"**/*{ext}"
        else:
            pattern = f"*{ext}"
        
        files = list(directory.glob(pattern))
        audio_files.extend(files)
    
    # Remove duplicates and sort
    audio_files = sorted(set(audio_files))
    
    if not audio_files:
        print(f"\n✗ No audio files found in {directory}")
        return {
            'total_files': 0,
            'total_duration_seconds': 0.0,
            'total_duration_formatted': '0 seconds',
            'average_duration_seconds': 0.0,
            'files_by_extension': {}
        }
    
    print(f"\nFound {len(audio_files)} audio file(s)")
    print("Calculating durations...")
    
    # Calculate durations
    total_duration = 0.0
    durations = []
    files_by_ext = {ext: [] for ext in extensions}
    failed_files = []
    
    iterator = tqdm(audio_files, desc="Processing files") if show_progress else audio_files
    
    for filepath in iterator:
        ext = filepath.suffix.lower()
        if ext in files_by_ext:
            files_by_ext[ext].append(filepath)
        
        duration = get_audio_duration(filepath)
        if duration > 0:
            total_duration += duration
            durations.append(duration)
        else:
            failed_files.append(filepath)
    
    # Calculate statistics
    avg_duration = sum(durations) / len(durations) if durations else 0.0
    min_duration = min(durations) if durations else 0.0
    max_duration = max(durations) if durations else 0.0
    
    # Count files by extension
    files_by_ext_count = {ext: len(files) for ext, files in files_by_ext.items()}
    
    results = {
        'total_files': len(audio_files),
        'successful_files': len(durations),
        'failed_files': len(failed_files),
        'total_duration_seconds': total_duration,
        'total_duration_formatted': format_duration(total_duration),
        'average_duration_seconds': avg_duration,
        'min_duration_seconds': min_duration,
        'max_duration_seconds': max_duration,
        'files_by_extension': files_by_ext_count,
        'failed_files_list': failed_files
    }
    
    return results


def print_summary(results: dict):
    """Print a formatted summary of the results."""
    print("\n" + "="*60)
    print("Audio Duration Summary")
    print("="*60)
    print(f"Total files found: {results['total_files']}")
    print(f"Successfully processed: {results['successful_files']}")
    
    if results['failed_files'] > 0:
        print(f"⚠ Failed to process: {results['failed_files']}")
    
    print(f"\nTotal duration: {results['total_duration_formatted']}")
    print(f"  ({results['total_duration_seconds']:.2f} seconds)")
    print(f"  ({results['total_duration_seconds'] / 3600:.2f} hours)")
    
    if results['successful_files'] > 0:
        print(f"\nAverage file duration: {format_duration(results['average_duration_seconds'])}")
        print(f"  Min: {format_duration(results['min_duration_seconds'])}")
        print(f"  Max: {format_duration(results['max_duration_seconds'])}")
    
    if results['files_by_extension']:
        print(f"\nFiles by extension:")
        for ext, count in sorted(results['files_by_extension'].items()):
            print(f"  {ext}: {count} file(s)")
    
    if results['failed_files_list']:
        print(f"\n⚠ Failed files:")
        for filepath in results['failed_files_list'][:10]:  # Show first 10
            print(f"  - {filepath}")
        if len(results['failed_files_list']) > 10:
            print(f"  ... and {len(results['failed_files_list']) - 10} more")


def main(directory: str, recursive: bool = True, extensions: list = None, 
         show_progress: bool = True):
    """
    Calculate and display total duration of audio files.
    
    Args:
        directory: Directory to search for audio files
        recursive: If True, search subdirectories recursively
        extensions: List of file extensions to include (default: ['.wav'])
        show_progress: If True, show progress bar
    """
    dir_path = Path(directory).expanduser().absolute()
    
    if not dir_path.exists():
        print(f"✗ Error: Directory does not exist: {dir_path}")
        return None
    
    if not dir_path.is_dir():
        print(f"✗ Error: Path is not a directory: {dir_path}")
        return None
    
    results = calculate_total_duration(
        dir_path, 
        recursive=recursive, 
        extensions=extensions,
        show_progress=show_progress
    )
    
    print_summary(results)
    
    return results


if __name__ == "__main__":
    parser = ArgumentParser(description="Calculate total duration of audio files in a directory")
    parser.add_argument(
        "-d", "--directory",
        required=True,
        type=str,
        help="Directory to search for audio files"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        default=True,
        help="Search subdirectories recursively (default: True)"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Do not search subdirectories"
    )
    parser.add_argument(
        "-e", "--extensions",
        nargs="+",
        default=['.wav'],
        help="File extensions to include (default: .wav). Can specify multiple: -e .wav .flac .mp3"
    )
    parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="show_progress",
        help="Do not show progress bar"
    )
    
    args = parser.parse_args()
    main(**args)
    
    """
    Usage examples:
    
    # Check all WAV files in a directory (recursive by default)
    python /mnt/c/Users/reiem/PythonProjects/Rec-RIR/dataset/check_speech_duration.py -d /mnt/c/Users/reiem/PythonProjects/Rec-RIR/data/speech
    
    # Check only in the specified directory (no subdirectories)
    python check_speech_duration.py -d /path/to/speech/data --no-recursive
    
    # Check multiple file types
    python check_speech_duration.py -d /path/to/speech/data -e .wav .flac .mp3
    
    # Check without progress bar
    python check_speech_duration.py -d /path/to/speech/data --no-progress
    """

