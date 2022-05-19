

from models import TransformerNet
from utils import *
import torch
from torch.autograd import Variable
import argparse
import os
from torchvision.utils import save_image
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to image")
    parser.add_argument("--checkpoint_model", type=str, required=True, help="Path to checkpoint model")
    args = parser.parse_args()
    print(args)
    # 这上面是读取命令行传入的参数，传入参数应对应接受的参数。

    os.makedirs("images/outputs", exist_ok=True)
    # 输出文件夹，如果不存在则创建之

#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # 如果电脑有编译好CUDA的GPU，则使用之。如果没有，就用CPU运行这个项目。

    transform = style_transform()
    # 这段代码是init图像预处理，归一化的函数。这段代码的名字和作用没有关系

    # Define model and load model checkpoint
    transformer = TransformerNet().to(device)
    # 这段代码是用Transformer实现的模型，模型需要在使用前实例化，然后传送到GPU（或者CPU，同上如果你电脑有GPU的话）。
    transformer.load_state_dict(torch.load(args.checkpoint_model, map_location='cpu'))
    # 从本地的训练好的模型中，加载模型的权重。
    transformer.eval()
    # 模型进入evaluation模式

    # Prepare input
    image_tensor = Variable(transform(Image.open(args.image_path))).to(device)
    image_tensor = image_tensor.unsqueeze(0)
    # 这一步就是用PIL库把本地的图片读进来，预处理归一化，转成tensor，送到GPU里（或者CPU，同上）

    # Stylize image
    # torch.no_grad的意思就是不记录梯度，被这句包裹住的语句全都不记录反向传播的梯度。
    with torch.no_grad():
        stylized_image = denormalize(transformer(image_tensor)).cpu() #在处理完之后把结果送回CPU。

    # 下面就是保存图片，生成一个路径，然后存里面。
    style = args.checkpoint_model.split("\\")[-1].split(".")[0]
    content = args.image_path.split("\\")[-1]
    save_image(stylized_image, f"images/outputs/stylized-{style}-{content}")
