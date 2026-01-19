# get git tags
import subprocess
import os
import requests


def download_file(url, destination):
    if os.path.exists(destination):
        print(f"{destination} already exists, skipping download.")
        return
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    with open(destination, "wb") as file:
        for data in response.iter_content(block_size):
            file.write(data)
    print(f"Downloaded {destination} from {url}")


def get_git_tags():
    result = subprocess.run(["git", "tag"], stdout=subprocess.PIPE)
    tags = result.stdout.decode().splitlines()
    tags = [tag for tag in tags if "RVE-2" in tag]  # only 2.x versions
    return tags


os.system("git clone https://github.com/TNTwise/real-video-enhancer.git")
os.chdir("real-video-enhancer")
download_file(
    "https://v.animethemes.moe/JujutsuKaisenS2-OP1-NCBD1080.webm", "test_video.webm"
)
download_file(
    "https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/rife4.6.pkl",
    "rife4.6.pkl",
)
tags = get_git_tags()  # reverse to get the latest version first
print("Found tags:", tags)
for tag in reversed(tags):
    os.system(
        "python3 backend/rve-backend.py -i test_video.webm --benchmark --interpolate_model rife4.6.pkl --interpolate_factor 2 -b tensorrt"
    )
    os.system(f"git checkout {tag}")
