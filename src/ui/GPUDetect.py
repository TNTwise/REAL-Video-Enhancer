try:
    from ..constants import PLATFORM 
    from ..Util import log
except Exception:
    PLATFORM = 'win32'
    def log(msg): print(msg)
import subprocess
import re

class GPUDetect:
    def __init__(self):
        self.gpu_info = self.get_gpu_info()

    def get_gpu_info(self):
        if PLATFORM == "win32":
            try:
                output = subprocess.check_output(
                    "nvidia-smi", shell=True
                ).decode()
                return str(output.strip().split("\n"))
            except Exception:
                return "Unable to retrieve GPU info on Windows"

        elif PLATFORM == "darwin":  # macOS
            try:
                output = subprocess.check_output(
                    "system_profiler SPDisplaysDataType | grep Vendor", shell=True
                ).decode()
                return output.strip().split(":")[1].strip()
            except Exception:
                return "Unable to retrieve GPU info on macOS"

        elif PLATFORM == "linux":
            try:
                # Try lspci command first
                output = subprocess.check_output("lspci | grep -i vga", shell=True).decode()
                return output.strip().split(":")[2].strip()
            except Exception:
                try:
                    # If lspci fails, try reading from /sys/class/graphics
                    with open("/sys/class/graphics/fb0/device/vendor", "r") as f:
                        vendor_id = f.read().strip()
                    return f"Vendor ID: {vendor_id}"
                except Exception:
                    return "Unable to retrieve GPU info on Linux"

        else:
            return "Unsupported operating system"


    def getVendor(self):
        """
        Gets GPU vendor of the system
        vendors = ["Intel", "AMD", "Nvidia"]
        """
        vendors = ["Intel", "AMD", "Nvidia"]
        for vendor in vendors:
            print(self.gpu_info)
            if vendor.lower() in self.gpu_info.lower():
                return vendor
        return None
    
    def getModelOfGPU(self):
        vendor = self.getVendor()
        model = "0"
        if vendor == "Nvidia":
            try:
                model = re.findall(r"RTX \d\d\d\d", self.gpu_info)[0]
                log("GPU Model Found: " + vendor + " " + model)
            except Exception:
                log("Couldnt find gpu model, " + self.gpu_info)
        return model
    
    def getPyTorchFeatures(self) -> str | None:
        try:
            if int(self.getModelOfGPU()[4]) >= 2 and self.getVendor() == "Nvidia":
                return "cuda"
        except Exception:
            return None
        return None
    
if __name__ == '__main__':
    print(GPUDetect().getPyTorchFeatures())