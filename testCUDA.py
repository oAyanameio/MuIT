import torch
print(torch.cuda.is_available())  # 应返回True
import torch
print(torch.cuda.get_device_name(0))
if torch.cuda.is_available():
    print("CUDA版本:", torch.version.cuda)
else:
    print("未检测到CUDA")

import torch
torch.cuda.set_device(0)
