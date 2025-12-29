from ..constants import CPU_ARCH, PLATFORM, CWD
import os
import sys

download_path = os.path.join(CWD, "ffmpeg")



def download_ffmpeg():
    from .FileHandler import FileHandler
    link = "https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/"
    match PLATFORM:
        case "linux":
            link += "ffmpeg" if CPU_ARCH == "x86_64" else "ffmpeg-linux-arm64"
        case "win32":
            link += "ffmpeg.exe" if CPU_ARCH == "x86_64" else "ffmpeg-windows-arm64.exe"
        case "darwin":
            link += "ffmpeg-macos-bin" if CPU_ARCH == "x86_64" else "ffmpeg-macos-arm"
    
    try:
        import requests
        print("Downloading FFMpeg from " + link)
        
        response = requests.get(link, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  
        with open(download_path, "wb") as file:
            for data in response.iter_content(block_size):
                file.write(data)
        print("Download completed.")

        return download_path
    except Exception as e:
        print(f"Error downloading FFMpeg: {e}", file=sys.stderr)
        return None
