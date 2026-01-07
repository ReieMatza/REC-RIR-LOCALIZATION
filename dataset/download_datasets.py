#!/usr/bin/env python3
# @author: Auto-generated
# @description: Script to download EARS, VCTK, and DNS Challenge datasets

import sys
import subprocess
import urllib.request
import zipfile
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


def download_ears_dataset(output_dir: str):
    """Download EARS dataset from GitHub releases."""
    print("\n" + "="*60)
    print("Downloading EARS Dataset")
    print("="*60)
    
    ears_dir = Path(output_dir) / "EARS"
    ears_dir.mkdir(parents=True, exist_ok=True)
    
    # Download all 107 speaker files
    base_url = "https://github.com/facebookresearch/ears_dataset/releases/download/dataset/"
    
    failed_downloads = []
    for i in range(1, 108):
        speaker_id = f"p{i:03d}"
        zip_file = ears_dir / f"{speaker_id}.zip"
        
        if zip_file.exists():
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


def download_vctk_dataset(output_dir: str):
    """Download VCTK dataset from Hugging Face."""
    print("\n" + "="*60)
    print("Downloading VCTK Dataset")
    print("="*60)
    
    vctk_dir = Path(output_dir) / "VCTK"
    vctk_dir.mkdir(parents=True, exist_ok=True)
    
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


def download_dns_challenge(output_dir: str):
    """Download DNS Challenge dataset from GitHub."""
    print("\n" + "="*60)
    print("Downloading DNS Challenge Dataset")
    print("="*60)
    
    dns_dir = Path(output_dir) / "DNS-Challenge"
    dns_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if git is available
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Error: git is not installed. Please install git to download DNS Challenge.")
        return None
    
    repo_url = "https://github.com/microsoft/DNS-Challenge.git"
    repo_dir = dns_dir / "DNS-Challenge"
    
    if repo_dir.exists():
        print(f"Repository already exists at {repo_dir}")
        print("Pulling latest changes...")
        try:
            subprocess.run(
                ["git", "pull"],
                cwd=repo_dir,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError:
            print("⚠ Warning: Could not pull latest changes. Using existing repository.")
    else:
        print(f"Cloning {repo_url}...")
        try:
            subprocess.run(
                ["git", "clone", repo_url, str(repo_dir)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print("✓ Repository cloned successfully!")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error cloning DNS Challenge repository: {e}")
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


def main(output_dir: str, datasets: list = None):
    """
    Download specified datasets to the output directory.
    
    Args:
        output_dir: Directory where datasets will be downloaded
        datasets: List of datasets to download. Options: 'ears', 'vctk', 'dns'. 
                 If None, downloads all datasets.
    """
    output_path = Path(output_dir).expanduser().absolute()
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Dataset Download Script")
    print(f"{'='*60}")
    print(f"Output directory: {output_path}")
    
    if datasets is None:
        datasets = ['ears', 'vctk', 'dns']
    
    datasets = [d.lower() for d in datasets]
    
    results = {}
    
    if 'ears' in datasets:
        results['ears'] = download_ears_dataset(str(output_path))
    
    if 'vctk' in datasets:
        results['vctk'] = download_vctk_dataset(str(output_path))
    
    if 'dns' in datasets:
        results['dns'] = download_dns_challenge(str(output_path))
    
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
    
    args = parser.parse_args()
    main(**args)
    
    """
    Usage examples:
    
    # Download all datasets
    python download_datasets.py -o /path/to/datasets
    
    # Download only EARS and VCTK
    python download_datasets.py -o /path/to/datasets -d ears vctk
    
    # Download only DNS Challenge
    python download_datasets.py -o /path/to/datasets -d dns
    """

