from platform import python_version
import torch

print('Python Version: ', python_version())

print('PyTorch Version: ', torch.__version__)
print('CUDA Available: ', torch.cuda.is_available())
print('Device Count: ', torch.cuda.device_count())
print('Current Device: ', torch.cuda.current_device())
print('Device Name', torch.cuda.get_device_name(0))
