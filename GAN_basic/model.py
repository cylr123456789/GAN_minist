# 目的：生成新图像
# 方法：训练generator

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os

training_data = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor(),)

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

# 查看图片尺寸、数量


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

'''======================Generator=========================='''
class GANGenerator(nn.Module):
    """GAN生成器模型：使用多层感知器（MLP）结构，将输入噪声向量转换为生成的数据样本（通常为图像）"""
    def __init__(self, input_size: int, output_size: int, hidden_size: int):
        """
        :param input_size : 输入向量的维度——噪声维度
        :param output_size: 输出向量的维度
        :param hidden_size: 隐藏层的维度
        """
        super(GANGenerator, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_size, hidden_size),          # MLP?
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, output_size),
                                   nn.Tanh()
                                   )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)                           # x是一个随机噪声向量，经过生成器的MLP结构，输出生成的数据样本
        return x


'''======================Discriminator================'''
class GANDiscriminator(nn.Module):
    """GAN判别器模型：判别器接收一个数据样本，预测它为真实数据的概率"""
    def __init__(self, input_size: int, hidden_size: int):
        """
        :param input_size : 输入向量的维度
        :param hidden_size: 隐藏层的维度
        """
        super(GANDiscriminator, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_size, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, 1),
                                   nn.Sigmoid()                     # 将隐藏层的输出映射到一个【标量】，Sigmoid()将输出值压缩到 (0, 1) 的范围，输出值表示输入数据为【真实数据】的概率
                                   )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x



# 创建G、D实例
generator = GANGenerator(100, 784, 256).to(device)
discriminator = GANDiscriminator(784, 256).to(device)

# 二分类交叉熵损失函数
criterion = nn.BCELoss()

# G、D参数——Adam优化器
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)

# 保存生成的样本
def save_samples(generator, device, n_samples=64):



    # Step 1: 随机生成噪声向量
    # 生成一个形状为 (n_samples, 100) 的张量 z，作为输入给生成器，用于生成样本
    # torch.randn 生成【正态分布】的随机数，这里生成 n_samples 个样本，每个样本有 100 维
                                                                    # GANGenerator(100, 784, 256)要求输入是100维的向量
    z = torch.randn(n_samples, 100).to(device)      # n_samples :生成的样本数量

    # Step 2: 使用生成器生成样本
    # 将噪声 z 输入到生成器中，生成器会输出对应的生成样本
    # 输出 samples 的形状为 (n_samples, output_size)，其中 output_size 是生成器输出的向量维度
    samples = generator(z)                          # 生成的样本数据，形状为 (n_samples, output_size)，其中 output_size 是生成器的输出向量的维度

    # Step 3: 调整生成样本的形状，准备用于显示
    # 使用 view 方法将 samples 的形状调整为 (n_samples, 1, 28, 28)
    # - n_samples 表示样本的数量
    # - 1 表示通道数 (灰度图像只有一个通道)
    # - 28x28 表示每张图片的高度和宽度
    # .cpu() 将数据从 GPU 转移到 CPU，
    # .detach() 则是切断该张量的反向传播图，以便显示
    samples = samples.view(n_samples, 1, 28, 28).cpu().detach()

    # Step 4: 创建样本的网格并显示
    # vutils.make_grid 会将多个样本整合成一个网格，方便可视化
    # - normalize=True 用于将样本的值范围归一化到 [0,1]
    # - nrow=8 设置网格的每行样本数为 8
    grid = vutils.make_grid(samples, normalize=True, nrow=8)

    # Step 5: 使用 Matplotlib 显示生成的网格图像
    # permute(1, 2, 0) 将图像维度从 (通道, 高度, 宽度) 转换为 (高度, 宽度, 通道)，以便 Matplotlib 正确显示
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()




'''===============================adversial training process==================================='''
def train_gan(generator, discriminator, train_dataloader, criterion, optimizer_g, optimizer_d, device, n_epochs):

    # 轮次
    for epoch in range(n_epochs):

        # 遍历每个 batch，从训练数据加载器中获取数据
        for i, data in enumerate(train_dataloader):
            real_data, _ = data                 # data 包含了 (real_data, _)，其中 real_data 是真实的图像数据
            batch_size = real_data.size(0)      # 获取当前 batch 的大小

            real_data  = real_data.view(batch_size, -1).to(device)

            # 创建标签，用于判别器的损失计算
            label_real = torch.ones(batch_size, 1).to(device)       # 真实  标签为 1
            label_fake = torch.zeros(batch_size, 1).to(device)      # 假样本标签为 0

            z = torch.randn(batch_size, 100).to(device)
            fake_data = generator(z)

            optimizer_d.zero_grad()
            output_real = discriminator(real_data)
            loss_real = criterion(output_real, label_real)
            loss_real.backward()

            output_fake = discriminator(fake_data.detach())
            loss_fake = criterion(output_fake, label_fake)
            loss_fake.backward()

            loss_d = loss_real + loss_fake          # 真实样本损失 + 生成样本损失
            optimizer_d.step()

            # 训练生成器
            z = torch.randn(batch_size, 100).to(device)
            fake_data = generator(z)

            optimizer_g.zero_grad()
            output_g = discriminator(fake_data)
            loss_g = criterion(output_g, label_real)
            loss_g.backward()

            optimizer_g.step()

            if i % 100 == 0:
                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, n_epochs, i, len(train_dataloader), loss_d.item(), loss_g.item()))

        # 每个 epoch 结束后保存生成的样本图像
        save_samples(generator, device)


train_gan(generator, discriminator, train_dataloader, criterion, optimizer_g, optimizer_d, device, n_epochs=2)

# 训练完之后，如何调用进行图像生成？