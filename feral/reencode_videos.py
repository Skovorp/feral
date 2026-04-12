#!/usr/bin/env python3
"""
FERAL Video Re-encoding Script with Auto-FFmpeg Installation
Converts videos to the required format for FERAL processing.

Features:
- Automatically downloads and sets up FFmpeg if not installed
- Cross-platform support (Windows, macOS, Linux)
- Parallel processing for faster conversion
- Progress tracking and error handling

Usage:
    python reencode_videos.py /path/to/input/videos /path/to/output/videos

Requirements:
    - Python 3.6+
    - Internet connection (for FFmpeg auto-download if needed)
"""

import os
import sys
import subprocess
import argparse
import platform
import urllib.request
import zipfile
import tarfile
from multiprocessing import Pool
import mimetypes
from pathlib import Path

# FFmpeg download URLs for different platforms
FFMPEG_URLS = {
    'Windows': 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip',
    'Darwin': 'https://evermeet.cx/ffmpeg/ffmpeg-7.1.zip',
    'Linux': 'https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz'
}

def get_platform():
    import platform
    sysname = platform.system()
    if sysname in ('Darwin', 'Windows'):
        return sysname
    else:
        return 'Linux'

def download_file(url, filename):
    """Download a file with progress indication."""
    print(f"Downloading {filename}...")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) // total_size)
            print(f"\rProgress: {percent}% ({downloaded}/{total_size} bytes)", end='')
    
    urllib.request.urlretrieve(url, filename, reporthook=progress_hook)
    print("\nDownload complete!")

def extract_ffmpeg(archive_path, extract_dir):
    """Extract FFmpeg from downloaded archive."""
    print(f"Extracting {archive_path}...")
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    elif archive_path.endswith('.tar.xz'):
        with tarfile.open(archive_path, 'r:xz') as tar_ref:
            tar_ref.extractall(extract_dir)
    
    print("Extraction complete!")

def find_ffmpeg_binary(extract_dir, platform_name):
    """Find the FFmpeg binary in the extracted directory."""
    if platform_name == 'Windows':
        # Look for ffmpeg.exe in subdirectories
        for root, dirs, files in os.walk(extract_dir):
            if 'ffmpeg.exe' in files:
                return os.path.join(root, 'ffmpeg.exe')
    else:
        # Look for ffmpeg binary in subdirectories  
        for root, dirs, files in os.walk(extract_dir):
            if 'ffmpeg' in files:
                ffmpeg_path = os.path.join(root, 'ffmpeg')
                # Make sure it's executable
                os.chmod(ffmpeg_path, 0o755)
                return ffmpeg_path
    
    return None

def setup_ffmpeg():
    """Download and setup FFmpeg if not available."""
    platform_name = get_platform()
    
    # Check if ffmpeg is already available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✅ FFmpeg is already installed and available!")
        return 'ffmpeg'
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FFmpeg not found in PATH. Setting up FFmpeg automatically...")
    
    # Create ffmpeg directory
    ffmpeg_dir = os.path.join(os.getcwd(), 'ffmpeg_portable')
    os.makedirs(ffmpeg_dir, exist_ok=True)
    
    # Check if we already have a portable version
    existing_ffmpeg = find_ffmpeg_binary(ffmpeg_dir, platform_name)
    if existing_ffmpeg and os.path.exists(existing_ffmpeg):
        print("✅ Using existing portable FFmpeg installation!")
        return existing_ffmpeg
    
    # Download FFmpeg
    if platform_name not in FFMPEG_URLS:
        raise RuntimeError(f"Unsupported platform: {platform_name}")
    
    url = FFMPEG_URLS[platform_name]
    archive_name = url.split('/')[-1]
    archive_path = os.path.join(ffmpeg_dir, archive_name)
    
    try:
        download_file(url, archive_path)
        extract_ffmpeg(archive_path, ffmpeg_dir)
        
        # Find the binary
        ffmpeg_binary = find_ffmpeg_binary(ffmpeg_dir, platform_name)
        
        if not ffmpeg_binary:
            raise RuntimeError("Could not find FFmpeg binary after extraction")
        
        # Test the binary
        subprocess.run([ffmpeg_binary, '-version'], capture_output=True, check=True)
        print("✅ FFmpeg setup complete!")
        
        # Clean up archive
        os.remove(archive_path)
        
        return ffmpeg_binary
        
    except Exception as e:
        print(f"❌ Error setting up FFmpeg: {e}")
        print("Please install FFmpeg manually from https://ffmpeg.org/download.html")
        sys.exit(1)

def is_video_file(filepath):
    """Check if file is a video based on MIME type."""
    mime_type, _ = mimetypes.guess_type(filepath)
    return mime_type is not None and mime_type.startswith('video')

def process_file(args):
    """Process a single video file with ffmpeg."""
    input_path, output_dir, ffmpeg_binary = args
    
    # Create output filename with .mp4 extension
    input_name = Path(input_path).stem
    output_path = os.path.join(output_dir, f"{input_name}.mp4")
    
    # Skip if output already exists
    if os.path.exists(output_path):
        print(f"Output already exists, skipping: {output_path}")
        return True
    
    # FFmpeg command for FERAL-compatible encoding
    cmd = [
        ffmpeg_binary, "-i", input_path,
        "-vf", "scale=256:256:flags=lanczos",  # Resize to 256x256 with high-quality scaling
        "-c:v", "libx264",                      # H.264 video codec
        "-pix_fmt", "yuv420p",                  # Standard pixel format
        "-crf", "25",                           # Constant rate factor (quality)
        "-preset", "superfast",                 # Encoding speed/quality balance
        "-an",                                  # Remove audio
        "-y",                                   # Overwrite output files
        output_path
    ]
    
    try:
        print(f"Processing: {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Successfully processed: {os.path.basename(output_path)}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error processing {input_path}: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Re-encode videos for FERAL processing with auto-FFmpeg setup")
    parser.add_argument("input_dir", help="Directory containing input videos")
    parser.add_argument("output_dir", help="Directory for re-encoded videos")
    parser.add_argument("--processes", "-p", type=int, default=4, 
                        help="Number of parallel processes (default: 4)")
    
    args = parser.parse_args()
    
    print("🎬 FERAL Video Re-encoding Script")
    print("=" * 50)
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"❌ Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    video_paths = []
    for filename in os.listdir(args.input_dir):
        filepath = os.path.join(args.input_dir, filename)
        if os.path.isfile(filepath) and is_video_file(filepath):
            video_paths.append(filepath)
        else:
            print(f"Input directory must only have videos. Found not video: {filepath}")
            sys.exit(1)
    if not video_paths:
        print("❌ No video files found in input directory.")
        sys.exit(1)
    print(f"📁 Found {len(video_paths)} video files to process")
    
    # Create output directory
    out_dir = Path(args.output_dir)
    if out_dir.exists():
        if any(out_dir.iterdir()):
            print(f"Directory '{out_dir}' should be empty")
            sys.exit(1)
    else:
        out_dir.mkdir(parents=True)
    
    # Setup FFmpeg (download if needed)
    ffmpeg_binary = setup_ffmpeg()
    input_files = [(x, args.output_dir, ffmpeg_binary) for x in video_paths]
    
    print(f"Using this ffmpeg path: {ffmpeg_binary}")
    print(f"Using {args.processes} parallel processes")
    print(f"Output directory: {args.output_dir}")
    print("-" * 50)
    
    # Process files in parallel
    with Pool(processes=args.processes) as pool:
        results = pool.map(process_file, input_files)
    
    successful = sum(results)
    total = len(input_files)
    print("-" * 50)
    print(f"🎉 Processing complete: {successful}/{total} files successful")
    
    if successful < total:
        print(f"⚠️  {total - successful} files failed to process")
        sys.exit(1)
    else:
        print("✨ All videos successfully re-encoded for FERAL!")
        print(f"📂 Converted videos are in: {args.output_dir}")

if __name__ == "__main__":
    main()