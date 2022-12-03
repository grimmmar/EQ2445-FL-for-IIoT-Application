import torch

print(torch.cuda.device_count())
print(torch.cuda.is_available())
print(torch.backends.cudnn.is_available())
print(torch.cuda_version)
print(torch.backends.cudnn.version())


print(torch.__version__)
print(torch.cuda.is_available())
