
import argparse
import os
import sys
import random
from PIL import Image
import numpy as np
import torch
import glob
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image
from models import TransformerNet, VGG16
from utils import *

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser for Fast-Neural-Style")
    parser.add_argument("--dataset_path", type=str, default='', help="path to training dataset")
    parser.add_argument("--style_image", type=str, default="style-images/mosaic.jpg", help="path to style image")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--image_size", type=int, default=256, help="Size of training images")
    parser.add_argument("--style_size", type=int, help="Size of style image")
    parser.add_argument("--lambda_content", type=float, default=1e5, help="Weight for content loss")
    parser.add_argument("--lambda_style", type=float, default=1e10, help="Weight for style loss")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--checkpoint_model", type=str, help="Optional path to checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=2000, help="Batches between saving model")
    parser.add_argument("--sample_interval", type=int, default=1000, help="Batches between saving image samples")
    args = parser.parse_args()
    # 这上面是读取命令行传入的参数，传入参数应对应接受的参数。

    style_name = args.style_image.split("\\")[-1].split(".")[0]
    os.makedirs(f"images/outputs/{style_name}-training", exist_ok=True)
    os.makedirs(f"checkpoints", exist_ok=True)
    # 输出文件夹，如果不存在则创建之
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   # 如果电脑有编译好CUDA的GPU，则使用之。如果没有，就用CPU运行这个项目。
    train_dataset = datasets.ImageFolder(args.dataset_path, train_transform(args.image_size))
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    # 这里是定义dataloader，说白了就是定义训练过程中读取数据集的方法和速度。
    # 关于dataloader是干什么的参考https://pytorch.org/docs/stable/data.html。
    
    # 定义模型，这段代码是用Transformer实现的模型，模型需要在使用前实例化，然后传送到GPU（或者CPU，同上如果你电脑有GPU的话）。
    transformer = TransformerNet().to(device)
    # Transformer前用VGG16提取特征。所以也要初始化VGG模型，所以这里是不需要记录VGG的梯度的，只需要用它推理就好了。
    vgg = VGG16(requires_grad=False).to(device)

    # Load checkpoint model if specified
    # 从本地的训练好的模型中，加载模型的权重。
    if args.checkpoint_model:
        transformer.load_state_dict(torch.load(args.checkpoint_model))

    # 定义 优化器 和损失函数，这里用了Adam优化器和MSELoss，关于MSE的理解:https://blog.csdn.net/weixin_38145317/article/details/103735784
    optimizer = Adam(transformer.parameters(), args.lr)#优化器
    l2_loss = torch.nn.MSELoss().to(device)#损失函数

    # 用PIL库把本地的目标风格图片读进来，预处理归一化，转成tensor，送到GPU里（或者CPU，同上）
    style = style_transform(args.style_size)(Image.open(args.style_image))
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)

    # 把上面风格图片的特征提出来，就是我们想学习的特征。
    features_style = vgg(style)
    gram_style = [gram_matrix(y) for y in features_style]

    # 随机提取 8 幅图像用于模型的视觉评估，通俗来讲就是看看效果怎么样
    image_samples = []
    for path in random.sample(glob.glob(f"{args.dataset_path}/*/*.jpg"), 8):
        # image_samples += [style_transform(args.image_size)(Image.open(path))]
        image_samples += [train_transform(args.image_size)(Image.open(path))]
    image_samples = torch.stack(image_samples)

    # 把随机提取的 8 幅图像，做一下视觉迁移，保存，看看效果。
    def save_sample(batches_done):
        """ Evaluates the model and saves image samples """
        transformer.eval()
        with torch.no_grad():
            output = transformer(image_samples.to(device))
        image_grid = denormalize(torch.cat((image_samples.cpu(), output.cpu()), 2))
        save_image(image_grid, f"images/outputs/{style_name}-training/{batches_done}.jpg", nrow=4)
        transformer.train()

    # 对每个epoch
    for epoch in range(args.epochs):
        # 这里是定义了一些metrics，实际上是风格迁移从各个方向定义的loss，定义了视觉迁移的效果。
        epoch_metrics = {"content": [], "style": [], "total": []}
        for batch_i, (images, _) in enumerate(dataloader):
            # 从dataloader中按batch抽取图像数据集。
            # 在训练期间将梯度定义为0，关于为什么这么做：
            # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
            optimizer.zero_grad()

            # 显卡和CPU是不能互访的，所以既然你把模型传到了其中某个里面，所有的数据也都要传过去。
            images_original = images.to(device)
            # 获取transform之后的图像
            images_transformed = transformer(images_original)

            # 用VGG提取原图和transform之后的特征，用来计算loss。这一步是用来防止图像“过于”风格化
            features_original = vgg(images_original)
            features_transformed = vgg(images_transformed)

            # Compute content loss as MSE between features
            content_loss = args.lambda_content * l2_loss(features_transformed.relu2_2, features_original.relu2_2)

            # 计算transformer模型输出的图像和风格图像的风格特征是否相近的loss
            style_loss = 0
            for ft_y, gm_s in zip(features_transformed, gram_style):  #(1,2,3) (4,5,6)  -> zip -> (1,4)(2,5)(3,6)
                gm_y = gram_matrix(ft_y)
                style_loss += l2_loss(gm_y, gm_s[: images.size(0), :, :])
            style_loss *= args.lambda_style

            # 这俩loss加起来，作为总共的loss（既关注，又关注）。
            total_loss = content_loss + style_loss

            # 依照loss进行方向传播
            total_loss.backward()
            # 学习率也要随着step下降
            optimizer.step()

            # 下面没什么好说的，把loss打印到屏幕上，规范化输出之类的。
            epoch_metrics["content"] += [content_loss.item()]
            epoch_metrics["style"] += [style_loss.item()]
            epoch_metrics["total"] += [total_loss.item()]

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Content: %.2f (%.2f) Style: %.2f (%.2f) Total: %.2f (%.2f)]"
                % (
                    epoch + 1,
                    args.epochs,
                    batch_i,
                    (len(train_dataset) / args.batch_size),
                    content_loss.item(),
                    np.mean(epoch_metrics["content"]),
                    style_loss.item(),
                    np.mean(epoch_metrics["style"]),
                    total_loss.item(),
                    np.mean(epoch_metrics["total"]),
                )
            )

            batches_done = epoch * len(dataloader) + batch_i + 1
            # 每隔几步看一下效果咋样（对sample的8张图推理看看效果）
            if batches_done % args.sample_interval == 0:
                save_sample(batches_done)
            
            # 每隔几步保存一下checkpoint
            if args.checkpoint_interval > 0 and batches_done % args.checkpoint_interval == 0:
                style_name = os.path.basename(args.style_image).split(".")[0]
                torch.save(transformer.state_dict(), f"checkpoints/{style_name}_{batches_done}.pth")
