import torch
print(torch.cuda.is_available())  # 应返回True
import torch
print(torch.cuda.get_device_name(0))

import torch
torch.cuda.set_device(0)