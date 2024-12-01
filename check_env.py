#加载torch
import torch
print("cudnn版本:",torch.backends.cudnn.version())
#输出8200，代表着成功安装了cudnn v8.4.0
print("torch版本:",torch.__version__)
#输出1.11.0，代表成功安装了pytorch 1.11.0
print("cuda版本:",torch.version.cuda)
#输出11.3，代表成功安装了cuda 11.3
print("gpu是否可用",torch.cuda.is_available())
#True