#!/usr/bin/env python3
# @author: Auto-generated
# @description: Script to download EARS, VCTK, and DNS Challenge datasets

import sys
import subprocess
import urllib.request
import zipfile
import shutil
from pathlib import Path
from jsonargparse import ArgumentParser


def download_file(url: str, output_path: str, desc: str = None):
    """Download a file with progress bar."""
    try:
        def reporthook(blocknum, blocksize, totalsize):
            if totalsize > 0:
                percent = min(100, (blocknum * blocksize * 100) / totalsize)
                sys.stdout.write(f"\r{desc or 'Downloading'}: {percent:.1f}%")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, output_path, reporthook)
        sys.stdout.write("\n")
        return True
    except Exception as e:
        print(f"\nError downloading {url}: {e}")
        return False


def check_ears_health(ears_dir: Path) -> bool:
    """Check if EARS dataset is already downloaded and healthy."""
    if not ears_dir.exists():
        return False
    
    # Check for at least some speaker directories (expect 107 speakers)
    speaker_dirs = [d for d in ears_dir.iterdir() if d.is_dir() and d.name.startswith('p')]
    
    if len(speaker_dirs) < 50:  # Require at least 50 speakers to consider it healthy
        return False
    
    # Check if speaker directories contain audio files
    audio_extensions = {'.wav', '.flac', '.mp3'}
    audio_files_found = 0
    for speaker_dir in speaker_dirs[:10]:  # Sample first 10 speakers
        audio_files = [f for f in speaker_dir.rglob('*') 
                      if f.is_file() and f.suffix.lower() in audio_extensions]
        if audio_files:
            audio_files_found += 1
    
    # If at least 8 out of 10 sampled speakers have audio files, consider it healthy
    if audio_files_found < 8:
        return False
    
    # Check for metadata files (optional but good to have)
    metadata_files = ['speaker_statistics.json', 'transcripts.json']
    metadata_count = sum(1 for f in metadata_files if (ears_dir / f).exists())
    
    # Dataset is healthy if we have speakers with audio files
    return True


def check_vctk_health(vctk_dir: Path) -> bool:
    """Check if VCTK dataset is already downloaded and healthy."""
    if not vctk_dir.exists():
        return False
    
    # Check if it's a git repository (from git clone)
    if (vctk_dir / ".git").exists():
        # Check if repository has content
        if any(vctk_dir.iterdir()):
            return True
        return False
    
    # Check if it's a HuggingFace dataset (has data files)
    # HuggingFace datasets typically have these structures:
    # - data-* files or arrow files
    # - dataset_info.json
    # - Or a data/ subdirectory
    
    data_files = list(vctk_dir.glob("data-*"))
    arrow_files = list(vctk_dir.glob("*.arrow"))
    dataset_info = vctk_dir / "dataset_info.json"
    data_dir = vctk_dir / "data"
    
    # Check for any of these indicators
    if data_files or arrow_files or dataset_info.exists() or (data_dir.exists() and any(data_dir.iterdir())):
        return True
    
    # Check for speaker directories (VCTK structure)
    speaker_dirs = [d for d in vctk_dir.iterdir() if d.is_dir() and d.name.startswith('p')]
    if len(speaker_dirs) > 10:  # VCTK has many speakers
        # Check if they contain audio files
        audio_extensions = {'.wav', '.flac'}
        for speaker_dir in speaker_dirs[:5]:  # Sample first 5
            audio_files = [f for f in speaker_dir.rglob('*') 
                          if f.is_file() and f.suffix.lower() in audio_extensions]
            if audio_files:
                return True
    
    return False


def check_dns_health(dns_dir: Path) -> bool:
    """Check if DNS Challenge repository is already downloaded and healthy."""
    if not dns_dir.exists():
        return False
    
    # Check if it's a git repository
    if (dns_dir / ".git").exists():
        # Check if repository has content
        files = list(dns_dir.iterdir())
        # Should have at least README, download scripts, or other files
        if len(files) > 3:  # .git, .gitignore, README, and more
            return True
    
    return False


def download_ears_dataset(output_dir: str, force: bool = False):
    """Download EARS dataset from GitHub releases."""
    print("\n" + "="*60)
    print("Downloading EARS Dataset")
    print("="*60)
    
    ears_dir = Path(output_dir) / "EARS"
    ears_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset is already downloaded and healthy
    if not force and check_ears_health(ears_dir):
        print(f"✓ EARS dataset already exists and appears healthy at {ears_dir}")
        print("  Skipping download. Use --force to re-download.")
        return ears_dir
    
    # If force mode, we still check for individual files to skip
    # Download all 107 speaker files
    base_url = "https://github.com/facebookresearch/ears_dataset/releases/download/dataset/"
    
    failed_downloads = []
    for i in range(1, 108):
        speaker_id = f"p{i:03d}"
        zip_file = ears_dir / f"{speaker_id}.zip"
        speaker_dir = ears_dir / speaker_id
        
        # Skip if already extracted (directory exists with content)
        if speaker_dir.exists() and any(speaker_dir.iterdir()):
            if not force:
                print(f"Skipping {speaker_id} (already extracted)")
                continue
        # Skip if zip exists and we're not forcing
        elif zip_file.exists() and not force:
            print(f"Skipping {speaker_id}.zip (already exists)")
            continue
        
        url = f"{base_url}{speaker_id}.zip"
        print(f"\nDownloading {speaker_id}.zip ({i}/107)...")
        
        if download_file(url, str(zip_file), f"{speaker_id}.zip"):
            # Extract the zip file
            print(f"Extracting {speaker_id}.zip...")
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(ears_dir)
                # Remove zip file after extraction
                zip_file.unlink()
                print(f"✓ Successfully downloaded and extracted {speaker_id}")
            except Exception as e:
                print(f"✗ Error extracting {speaker_id}.zip: {e}")
                failed_downloads.append(speaker_id)
        else:
            failed_downloads.append(speaker_id)
    
    # Download additional files
    additional_files = {
        "speaker_statistics.json": "https://raw.githubusercontent.com/facebookresearch/ears_dataset/main/speaker_statistics.json",
        "transcripts.json": "https://raw.githubusercontent.com/facebookresearch/ears_dataset/main/transcripts.json"
    }
    
    for filename, url in additional_files.items():
        filepath = ears_dir / filename
        if not filepath.exists():
            print(f"\nDownloading {filename}...")
            download_file(url, str(filepath), filename)
    
    if failed_downloads:
        print(f"\n⚠ Warning: Failed to download {len(failed_downloads)} files: {failed_downloads}")
    else:
        print("\n✓ EARS dataset download completed successfully!")
    
    return ears_dir


def download_vctk_dataset(output_dir: str, force: bool = False):
    """Download VCTK dataset from Hugging Face."""
    print("\n" + "="*60)
    print("Downloading VCTK Dataset")
    print("="*60)
    
    vctk_dir = Path(output_dir) / "VCTK"
    vctk_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset is already downloaded and healthy
    if not force and check_vctk_health(vctk_dir):
        print(f"✓ VCTK dataset already exists and appears healthy at {vctk_dir}")
        print("  Skipping download. Use --force to re-download.")
        return vctk_dir
    
    # Check if git is available
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Error: git is not installed. Please install git to download VCTK dataset.")
        print("Alternatively, you can manually download from: https://huggingface.co/datasets/confit/vctk-full")
        return None
    
    # Try using huggingface_hub or datasets library, otherwise use git
    try:
        try:
            from datasets import load_dataset  # type: ignore
            print("Using Hugging Face datasets library...")
            print("Loading VCTK dataset (this may take a while)...")
            
            # Load dataset
            dataset = load_dataset("confit/vctk-full", split="train")
            
            # Save to local directory
            print("Saving dataset to local directory...")
            dataset.save_to_disk(str(vctk_dir))
            
            print("✓ VCTK dataset download completed successfully!")
            return vctk_dir
        except ImportError:
            # Try using huggingface_hub directly
            try:
                from huggingface_hub import snapshot_download  # type: ignore
                print("Using Hugging Face Hub library...")
                print("Downloading VCTK dataset (this may take a while)...")
                
                snapshot_download(
                    repo_id="confit/vctk-full",
                    repo_type="dataset",
                    local_dir=str(vctk_dir),
                    local_dir_use_symlinks=False
                )
                
                print("✓ VCTK dataset download completed successfully!")
                return vctk_dir
            except ImportError:
                print("Hugging Face libraries not found. Using git clone instead...")
                print("Note: For better performance, install with: pip install datasets huggingface_hub")
                raise ImportError("No Hugging Face libraries available")
    except ImportError:
        # Clone using git
        repo_url = "https://huggingface.co/datasets/confit/vctk-full"
        print(f"Cloning {repo_url}...")
        print("Note: This may require git-lfs for large files. Install with: git lfs install")
        
        try:
            # Check if git-lfs is installed
            try:
                subprocess.run(["git", "lfs", "version"], check=True, capture_output=True)
                print("git-lfs is available.")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("⚠ Warning: git-lfs not found. Large files may not download correctly.")
                print("  Install with: git lfs install")
            
            subprocess.run(
                ["git", "clone", repo_url, str(vctk_dir)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print("✓ VCTK dataset download completed successfully!")
            return vctk_dir
        except subprocess.CalledProcessError as e:
            print(f"✗ Error cloning VCTK dataset: {e}")
            print("You may need to authenticate with Hugging Face. Try:")
            print("  git lfs install")
            print("  huggingface-cli login")
            return None
    except Exception as e:
        print(f"✗ Error downloading VCTK dataset: {e}")
        return None


def download_dns_challenge(output_dir: str, force: bool = False):
    """Download DNS Challenge dataset from GitHub."""
    print("\n" + "="*60)
    print("Downloading DNS Challenge Dataset")
    print("="*60)
    
    dns_dir = Path(output_dir) / "DNS-Challenge"
    dns_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset is already downloaded and healthy
    if not force and check_dns_health(dns_dir):
        print(f"✓ DNS Challenge repository already exists and appears healthy at {dns_dir}")
        print("  Skipping download. Use --force to re-download.")
        return dns_dir
    
    # Check if git is available
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Error: git is not installed. Please install git to download DNS Challenge.")
        return None
    
    repo_url = "https://github.com/microsoft/DNS-Challenge.git"
    # Use dns_dir directly as the clone target to avoid nested directories
    repo_dir = dns_dir
    
    # Check if .git directory exists (indicating it's already a git repo)
    if (repo_dir / ".git").exists():
        print(f"Repository already exists at {repo_dir}")
        print("Pulling latest changes...")
        try:
            result = subprocess.run(
                ["git", "pull"],
                cwd=repo_dir,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("✓ Repository updated successfully!")
            else:
                print("⚠ Warning: Could not pull latest changes. Using existing repository.")
        except Exception as e:
            print(f"⚠ Warning: Could not pull latest changes: {e}. Using existing repository.")
    else:
        # Check if directory exists and has files (but not a git repo)
        if repo_dir.exists() and any(repo_dir.iterdir()) and not (repo_dir / ".git").exists():
            print(f"⚠ Warning: Directory {repo_dir} exists but is not a git repository.")
            print("  Removing existing directory to allow fresh clone...")
            shutil.rmtree(repo_dir)
            repo_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Cloning {repo_url}...")
        print(f"Target directory: {repo_dir}")
        try:
            # Ensure parent directory exists
            repo_dir.parent.mkdir(parents=True, exist_ok=True)
            
            # Try shallow clone first (faster)
            print("Attempting shallow clone (depth=1)...")
            result = subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(repo_dir)],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                # If shallow clone fails, try full clone
                print("Shallow clone failed, trying full clone...")
                # Remove partial clone if it exists
                if repo_dir.exists():
                    shutil.rmtree(repo_dir)
                    repo_dir.mkdir(parents=True, exist_ok=True)
                
                result = subprocess.run(
                    ["git", "clone", repo_url, str(repo_dir)],
                    capture_output=True,
                    text=True
                )
            
            if result.returncode != 0:
                # Show the actual error
                error_msg = result.stderr if result.stderr else result.stdout
                print(f"✗ Error cloning DNS Challenge repository:")
                print(f"  {error_msg}")
                print(f"\n  Exit code: {result.returncode}")
                print("\n  Troubleshooting:")
                print("  - Check your internet connection")
                print("  - Verify the repository URL is correct")
                print("  - Try cloning manually: git clone https://github.com/microsoft/DNS-Challenge.git")
                print(f"  - Check if directory is writable: {repo_dir.parent}")
                return None
            
            print("✓ Repository cloned successfully!")
        except Exception as e:
            print(f"✗ Unexpected error cloning DNS Challenge repository: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # Check for download scripts
    download_scripts = list(repo_dir.glob("download*.sh"))
    download_scripts.extend(list(repo_dir.glob("download*.py")))
    
    if download_scripts:
        print(f"\nFound {len(download_scripts)} download script(s):")
        for script in download_scripts:
            print(f"  - {script.name}")
        print("\n⚠ Note: You may need to run the download script(s) manually to download the actual dataset files.")
        print(f"   Scripts are located in: {repo_dir}")
    else:
        print("\n⚠ Note: No download scripts found. Please check the repository for download instructions.")
    
    print("✓ DNS Challenge repository download completed!")
    return dns_dir


def main(output_dir: str, datasets: list = None, force: bool = False):
    """
    Download specified datasets to the output directory.
    
    Args:
        output_dir: Directory where datasets will be downloaded
        datasets: List of datasets to download. Options: 'ears', 'vctk', 'dns'. 
                 If None, downloads all datasets.
        force: If True, force re-download even if dataset exists and appears healthy
    """
    output_path = Path(output_dir).expanduser().absolute()
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Dataset Download Script")
    print(f"{'='*60}")
    print(f"Output directory: {output_path}")
    
    if force:
        print("⚠ Force mode: Will re-download existing datasets")
    
    if datasets is None:
        datasets = ['ears', 'vctk', 'dns']
    
    datasets = [d.lower() for d in datasets]
    
    results = {}
    
    if 'ears' in datasets:
        results['ears'] = download_ears_dataset(str(output_path), force=force)
    
    if 'vctk' in datasets:
        results['vctk'] = download_vctk_dataset(str(output_path), force=force)
    
    if 'dns' in datasets:
        results['dns'] = download_dns_challenge(str(output_path), force=force)
    
    # Summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    for dataset, path in results.items():
        if path:
            print(f"✓ {dataset.upper()}: {path}")
        else:
            print(f"✗ {dataset.upper()}: Failed")
    
    print(f"\nAll datasets saved to: {output_path}")
    return results


if __name__ == "__main__":
    parser = ArgumentParser(description="Download EARS, VCTK, and DNS Challenge datasets")
    parser.add_argument(
        "-o", "--output_dir",
        required=True,
        type=str,
        help="Output directory where datasets will be downloaded"
    )
    parser.add_argument(
        "-d", "--datasets",
        nargs="+",
        choices=['ears', 'vctk', 'dns'],
        default=None,
        help="Specific datasets to download (default: all)"
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Force re-download even if dataset already exists and appears healthy"
    )
    
    args = parser.parse_args()
    main(**args)
    
    """
    Usage examples:
    
    # Download all datasets (skips if already downloaded and healthy)
    python download_datasets.py -o /path/to/datasets
    
    # Download only EARS and VCTK
    python download_datasets.py -o /path/to/datasets -d ears vctk
    
    # Download only DNS Challenge
    python download_datasets.py -o /path/to/datasets -d dns
    
    # Force re-download even if datasets exist
    python download_datasets.py -o /path/to/datasets -f
    """

