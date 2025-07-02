import os
import subprocess
from multiprocessing import Pool

input_dir = "/mnt/aperto/peter/feral_data/calms/all_raw"
output_dir = "/mnt/aperto/peter/feral_data/calms/reencoded_512_sleap"
os.makedirs(output_dir, exist_ok=True)

def process_file(filename):
    if not filename.endswith(".mp4"):
        return
    input_path = os.path.join(input_dir, filename)
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
    with Pool(processes=16) as pool:  # adjust number of processes as needed
        pool.map(process_file, files)
