#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
import torch
from torchsummary import summary
import torchvision
from nets.frcnn import FasterRCNN

if __name__ == "__main__":
    # 需要使用device来指定网络在GPU还是CPU运行
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # vgg一共16个卷积层
    m       = FasterRCNN(20, backbone = 'vgg').to(device)
    summary(m, input_size=(3, 600, 600))

    # print(device)
    # print(torch.__version__)
    # print(torchvision.__version__)
