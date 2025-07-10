import os
import subprocess
from multiprocessing import Pool
import mimetypes

input_dir = "/mnt/aperto/peter/feral_data/mosquitos/videos_storage"
output_dir = "/mnt/aperto/peter/feral_data/mosquitos/reencoded"
os.makedirs(output_dir, exist_ok=True)

def is_video_file(filepath):
    mime_type, _ = mimetypes.guess_type(filepath)
    return mime_type is not None and mime_type.startswith('video')

def process_file(filename):
    input_path = os.path.join(input_dir, filename)
    assert is_video_file(input_path), f"this is not a video: {input_path}"
    filename = os.path.splitext(filename)[0] + '.mp4'
    output_path = os.path.join(output_dir, filename)

    cmd = [
        "ffmpeg", "-i", input_path,
        "-vf", "scale=512:512:flags=lanczos",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "25",
        "-preset", "superfast",
        "-an",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    files = os.listdir(input_dir)
    with Pool(processes=16) as pool:
        pool.map(process_file, files)
