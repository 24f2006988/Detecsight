import torch

print("torch version      :", torch.__version__)
print("compiled CUDA      :", torch.version.cuda)
print("CUDA available     :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("device name        :", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("total VRAM (GB)    :", round(props.total_memory / 1024**3, 2))
    print("compute capability :", f"{props.major}.{props.minor}")
else:
    print("No CUDA device. You installed the CPU-only wheel — reinstall from the cu126 index.")
