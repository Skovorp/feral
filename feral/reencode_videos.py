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
    """Return the normalized platform name: 'Darwin', 'Windows', or 'Linux' (the fallback for any other OS)."""
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
        """urlretrieve reporthook that prints download progress as a percentage."""
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

DEFAULT_SMALLEST_SIDE = 512

def build_scale_filter(smallest_side):
    """Build an ffmpeg scale filter that downsizes the video so its smallest side
    is at most `smallest_side`, preserving aspect ratio. Never upscales. Output
    dimensions are forced to multiples of 2 (required by libx264)."""
    # When iw > ih: height is smaller -> cap height at smallest_side, auto width.
    # When iw <= ih: width is smaller -> cap width at smallest_side, auto height.
    s = smallest_side
    return (
        f"scale='if(gt(iw,ih),-2,min({s},iw))'"
        f":'if(gt(iw,ih),min({s},ih),-2)'"
        ":flags=lanczos"
    )

def process_file(args):
    """Process a single video file with ffmpeg."""
    if len(args) == 4:
        input_path, output_dir, ffmpeg_binary, smallest_side = args
    else:
        input_path, output_dir, ffmpeg_binary = args
        smallest_side = DEFAULT_SMALLEST_SIDE

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
        "-vf", build_scale_filter(smallest_side),
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

if __name__ == "__main__":
    from feral.cli import main
    main()