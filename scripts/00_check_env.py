"""Print environment facts; warn on low disk."""
import os, platform, shutil, sys
from pathlib import Path
try:
    import psutil
except ImportError:
    psutil = None

def main():
    print("OS:", platform.platform()); print("Python:", sys.version.split()[0])
    try:
        import torch
        print("torch:", torch.__version__, "cuda_available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print("GPU:", torch.cuda.get_device_name(0), f"free {free/2**30:.2f} GiB / {total/2**30:.2f} GiB")
    except ImportError:
        print("torch: not installed")
    root = Path(__file__).resolve().parents[1]
    free_gb = shutil.disk_usage(root).free / 2**30
    print(f"free disk: {free_gb:.1f} GiB")
    if free_gb < 15: print("WARNING: <15 GB free -> strict disk strategy REQUIRED")
    if psutil:
        vm = psutil.virtual_memory(); print(f"RAM: total {vm.total/2**30:.1f} GiB, available {vm.available/2**30:.1f} GiB")
    print("CPU count:", os.cpu_count())
if __name__ == "__main__": main()
