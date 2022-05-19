# coding: utf-8

from models import TransformerNet
from utils import style_transform, denormalize, deprocess
import torch
from torch.autograd import Variable
import argparse
import os
from torchvision.utils import save_image
from PIL import Image
import numpy as np



def load_pth(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 如果电脑有编译好CUDA的GPU，则使用之。如果没有，就用CPU运行这个项目。

    transform = style_transform()
    # 这段代码是init图像预处理，归一化的函数。这段代码的名字和作用没有关系

    # Define model and load model checkpoint
    transformer = TransformerNet().to(device)
    # 这段代码是用Transformer实现的模型，模型需要在使用前实例化，然后传送到GPU（或者CPU，同上如果你电脑有GPU的话）。
    transformer.load_state_dict(torch.load(model_path, map_location='cpu'))
    return transformer

# 模型调用的主函数
def main(_, input_path, model_path):
    os.makedirs("./images/outputs", exist_ok=True)
    # 输出文件夹，如果不存在则创建之

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 如果电脑有编译好CUDA的GPU，则使用之。如果没有，就用CPU运行这个项目。

    transform = style_transform()
    # 这段代码是init图像预处理，归一化的函数。这段代码的名字和作用没有关系

    # Define model and load model checkpoint
    transformer = TransformerNet().to(device)
    # 这段代码是用Transformer实现的模型，模型需要在使用前实例化，然后传送到GPU（或者CPU，同上如果你电脑有GPU的话）。
    transformer.load_state_dict(torch.load(model_path, map_location='cpu'))
    # 从本地的训练好的模型中，加载模型的权重。
    transformer.eval()
    # 模型进入evaluation模式

    # Prepare input
    image_tensor = Variable(transform(Image.open(input_path))).to(device)
    image_tensor = image_tensor.unsqueeze(0)
    # 这一步就是用PIL库把本地的图片读进来，预处理归一化，转成tensor，送到GPU里（或者CPU，同上）

    with torch.no_grad():
        stylized_image = denormalize(transformer(image_tensor))  # 在处理完之后把结果送回CPU。
    generated_file = './images/outputs/res.jpg'
    save_image(stylized_image, generated_file)
    # output = deprocess(stylized_image)
    # output = Image.fromarray(np.uint8(output))
    # generated_file = './images/outputs/res.jpg'
    # output.save(generated_file)


    return generated_file

