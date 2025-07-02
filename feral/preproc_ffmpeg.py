import os
import subprocess
from multiprocessing import Pool

input_dir = "/mnt/aperto/peter/feral_data/calms/all_raw"
output_dir = "/mnt/aperto/peter/feral_data/calms/reencoded_orig_size"
os.makedirs(output_dir, exist_ok=True)

def process_file(filename):
    if not filename.endswith(".mp4"):
        return
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)
    cmd = [
        "ffmpeg", "-i", input_path,
        # "-vf", "scale=256:256:flags=lanczos",
        "-g", "30",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "slow",
        "-an",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    files = os.listdir(input_dir)
    with Pool(processes=16) as pool:  # adjust number of processes as needed
        pool.map(process_file, files)
