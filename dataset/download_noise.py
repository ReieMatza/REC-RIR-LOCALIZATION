#!/usr/bin/env python3
# @author: Auto-generated
# @description: Script to download NoiseX-92 dataset from GitHub

import sys
import subprocess
import shutil
from pathlib import Path
from jsonargparse import ArgumentParser


def download_noisex92(output_dir: str, use_sparse_checkout: bool = True):
    """
    Download NoiseX-92 dataset from https://github.com/speechdnn/Noises
    
    Args:
        output_dir: Directory where NoiseX-92 will be downloaded
        use_sparse_checkout: If True, only downloads NoiseX-92 folder (faster)
    """
    print("\n" + "="*60)
    print("Downloading NoiseX-92 Dataset")
    print("="*60)
    
    # Check if git is available
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Error: git is not installed. Please install git to download NoiseX-92.")
        return None
    
    output_path = Path(output_dir).expanduser().absolute()
    noisex92_dir = output_path / "NoiseX-92"
    
    # Check if already exists
    if noisex92_dir.exists() and any(noisex92_dir.iterdir()):
        print(f"✓ NoiseX-92 already exists at {noisex92_dir}")
        print("  Skipping download. Use --force to re-download.")
        return noisex92_dir
    
    repo_url = "https://github.com/speechdnn/Noises.git"
    temp_dir = output_path / "Noises_temp"
    
    try:
        # Remove temp directory if it exists
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        print(f"Cloning repository: {repo_url}")
        print("This may take a while depending on your internet connection...")
        
        if use_sparse_checkout:
            # Use sparse checkout to only download NoiseX-92 folder
            print("\nUsing sparse checkout to download only NoiseX-92 folder...")
            
            sparse_temp = output_path / "Noises_sparse_temp"
            sparse_temp.mkdir(exist_ok=True)
            
            try:
                # Initialize repository in temp directory
                subprocess.run(
                    ["git", "init"],
                    cwd=sparse_temp,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Enable sparse checkout
                subprocess.run(
                    ["git", "sparse-checkout", "init", "--cone"],
                    cwd=sparse_temp,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Set sparse checkout path
                subprocess.run(
                    ["git", "sparse-checkout", "set", "NoiseX-92"],
                    cwd=sparse_temp,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Add remote
                subprocess.run(
                    ["git", "remote", "add", "origin", repo_url],
                    cwd=sparse_temp,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Pull only the NoiseX-92 folder
                print("Downloading NoiseX-92 folder...")
                result = subprocess.run(
                    ["git", "pull", "origin", "master"],
                    cwd=sparse_temp,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    # Try main branch if master fails
                    print("Trying 'main' branch instead of 'master'...")
                    result = subprocess.run(
                        ["git", "pull", "origin", "main"],
                        cwd=sparse_temp,
                        capture_output=True,
                        text=True
                    )
                
                if result.returncode != 0:
                    raise subprocess.CalledProcessError(
                        result.returncode, 
                        "git pull",
                        result.stderr
                    )
                
                # Move NoiseX-92 to final location
                downloaded_noisex92 = sparse_temp / "NoiseX-92"
                if downloaded_noisex92.exists():
                    if noisex92_dir.exists():
                        shutil.rmtree(noisex92_dir)
                    shutil.move(str(downloaded_noisex92), str(noisex92_dir))
                    
                    print(f"✓ NoiseX-92 dataset downloaded successfully!")
                    print(f"  Location: {noisex92_dir}")
                    return noisex92_dir
                else:
                    raise FileNotFoundError("NoiseX-92 folder not found after download")
            finally:
                # Clean up sparse temp directory
                if sparse_temp.exists():
                    shutil.rmtree(sparse_temp)
        
        else:
            # Full clone method (downloads entire repository)
            print("\nCloning entire repository (this may take longer)...")
            subprocess.run(
                ["git", "clone", repo_url, str(temp_dir)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Check if NoiseX-92 exists in the cloned repo
            source_noisex92 = temp_dir / "NoiseX-92"
            if not source_noisex92.exists():
                raise FileNotFoundError(
                    f"NoiseX-92 folder not found in repository at {source_noisex92}"
                )
            
            # Move NoiseX-92 to final location
            if noisex92_dir.exists():
                shutil.rmtree(noisex92_dir)
            
            shutil.move(str(source_noisex92), str(noisex92_dir))
            print(f"✓ NoiseX-92 dataset downloaded successfully!")
            print(f"  Location: {noisex92_dir}")
            
            return noisex92_dir
            
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error during git operation: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"  Error details: {e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr}")
        print("\nTrying alternative download method...")
        
        # Fallback: try full clone if sparse checkout failed
        if use_sparse_checkout:
            print("Attempting full repository clone instead...")
            try:
                # Clean up any partial downloads
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                if (output_path / ".git").exists():
                    shutil.rmtree(output_path / ".git")
                
                return download_noisex92(output_dir, use_sparse_checkout=False)
            except Exception as fallback_error:
                print(f"✗ Fallback method also failed: {fallback_error}")
                return None
        else:
            return None
    
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        return None
    
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return None
    
    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"⚠ Warning: Could not remove temporary directory {temp_dir}: {e}")


def download_all_noises(output_dir: str):
    """
    Download all noise datasets (NoiseX-92, Noise15, Nonspeech) from the repository.
    
    Args:
        output_dir: Directory where noise datasets will be downloaded
    """
    print("\n" + "="*60)
    print("Downloading All Noise Datasets")
    print("="*60)
    
    # Check if git is available
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Error: git is not installed. Please install git.")
        return None
    
    output_path = Path(output_dir).expanduser().absolute()
    repo_url = "https://github.com/speechdnn/Noises.git"
    temp_dir = output_path / "Noises_temp"
    
    try:
        # Remove temp directory if it exists
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        print(f"Cloning repository: {repo_url}")
        subprocess.run(
            ["git", "clone", repo_url, str(temp_dir)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Move each noise dataset
        noise_datasets = ["NoiseX-92", "Noise15", "Nonspeech"]
        results = {}
        
        for noise_name in noise_datasets:
            source_dir = temp_dir / noise_name
            target_dir = output_path / noise_name
            
            if source_dir.exists():
                if target_dir.exists():
                    print(f"⚠ {noise_name} already exists, skipping...")
                else:
                    shutil.move(str(source_dir), str(target_dir))
                    print(f"✓ {noise_name} downloaded successfully!")
                results[noise_name] = target_dir
            else:
                print(f"⚠ {noise_name} not found in repository")
                results[noise_name] = None
        
        return results
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error cloning repository: {e}")
        return None
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return None
    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"⚠ Warning: Could not remove temporary directory: {e}")


def main(output_dir: str, all_noises: bool = False, force: bool = False):
    """
    Download noise dataset(s) to the output directory.
    
    Args:
        output_dir: Directory where noise datasets will be downloaded
        all_noises: If True, download all noise datasets (NoiseX-92, Noise15, Nonspeech)
        force: If True, re-download even if dataset already exists
    """
    output_path = Path(output_dir).expanduser().absolute()
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Noise Dataset Download Script")
    print(f"{'='*60}")
    print(f"Output directory: {output_path}")
    
    if force:
        print("⚠ Force mode: Will re-download existing datasets")
    
    if all_noises:
        results = download_all_noises(str(output_path))
    else:
        noisex92_dir = output_path / "NoiseX-92"
        if not force and noisex92_dir.exists() and any(noisex92_dir.iterdir()):
            print(f"\n✓ NoiseX-92 already exists at {noisex92_dir}")
            print("  Use --force to re-download")
            results = {"NoiseX-92": noisex92_dir}
        else:
            if force and noisex92_dir.exists():
                print(f"Removing existing NoiseX-92 directory...")
                shutil.rmtree(noisex92_dir)
            
            result = download_noisex92(str(output_path))
            results = {"NoiseX-92": result} if result else {"NoiseX-92": None}
    
    # Summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    for name, path in results.items():
        if path:
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name}: Failed")
    
    print(f"\nAll noise datasets saved to: {output_path}")
    return results


if __name__ == "__main__":
    parser = ArgumentParser(description="Download NoiseX-92 and other noise datasets")
    parser.add_argument(
        "-o", "--output_dir",
        required=True,
        type=str,
        help="Output directory where noise datasets will be downloaded"
    )
    parser.add_argument(
        "-a", "--all_noises",
        action="store_true",
        help="Download all noise datasets (NoiseX-92, Noise15, Nonspeech)"
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Force re-download even if dataset already exists"
    )
    
    args = parser.parse_args()
    main(**args)
    
    """
    Usage examples:
    
    # Download only NoiseX-92
    python download_noise.py -o /mnt/c/Users/reiem/PythonProjects/Rec-RIR/data/noise
    
    # Download all noise datasets
    python download_noise.py -o /path/to/noise/data -a
    
    # Force re-download
    python download_noise.py -o /path/to/noise/data -f
    """

