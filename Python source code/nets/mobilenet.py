from torch import nn
from torch.hub import load_state_dict_from_url
import os

import torch
import torch.nn as nn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#inverted_residual_setting是一个列表，其中包含多个四元组（t, c, n, s），表示每个inverted residual block的一些参数。其中，t是扩展比例（expand_ratio），c是输出通道数（output_channel），n是该类型block的个数，s是步幅（stride）。
# _make_divisible是一个内部函数，将通道数c变为一个可以整除8的数。
# 代码首先对每个四元组调用_make_divisible函数计算输出通道数，然后用这个参数和其他参数初始化一个InvertedResidual block，将其添加到features列表中。这个列表最后被传递给nn.Sequential，形成特征提取部分的神经网络。
# 特征提取部分结束后，将其输出的tensor用ConvBNReLU进行降维，然后经过平均池化层，最后输入到全连接层中，输出一个大小为num_classes的向量。
# 代码最后通过循环遍历所有模块，初始化卷积层、批归一化层、线性层的参数。
# forward函数接受一个tensor作为输入，首先使用cbam模块处理，然后输入到特征提取部分进行特征提取，再经过降维、平均池化和全连接层，输出一个大小为num_classes的向量。（前向传播函数中，首先将输入通过cbam模块，然后经过feature部分，最后通过分类器输出结果）

class CBAM(nn.Module):
    def __init__(self, in_channel):
        super(CBAM, self).__init__()
        self.Cam = ChannelAttentionModul(in_channel=in_channel)  # 通道注意力模块
        self.Sam = SpatialAttentionModul(in_channel=in_channel)  # 空间注意力模块

    def forward(self, x):
        x = self.Cam(x)
        x = self.Sam(x)
        return x


class ChannelAttentionModul(nn.Module):  # 通道注意力模块
    def __init__(self, in_channel, r=0.5):  # channel为输入的维度, r为全连接层缩放比例->控制中间层个数
        super(ChannelAttentionModul, self).__init__()
        # 全局最大池化
        self.MaxPool = nn.AdaptiveMaxPool2d(1)

        self.fc_MaxPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),  # int(channel * r)取整数, 中间层神经元数至少为1, 如有必要可设为向上取整
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        # 全局均值池化
        self.AvgPool = nn.AdaptiveAvgPool2d(1)

        self.fc_AvgPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),  # int(channel * r)取整数, 中间层神经元数至少为1, 如有必要可设为向上取整
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1.最大池化分支
        max_branch = self.MaxPool(x)
        # 送入MLP全连接神经网络, 得到权重
        max_in = max_branch.view(max_branch.size(0), -1)
        max_weight = self.fc_MaxPool(max_in)

        # 2.全局池化分支
        avg_branch = self.AvgPool(x)
        # 送入MLP全连接神经网络, 得到权重
        avg_in = avg_branch.view(avg_branch.size(0), -1)
        avg_weight = self.fc_AvgPool(avg_in)

        # MaxPool + AvgPool 激活后得到权重weight
        weight = max_weight + avg_weight
        weight = self.sigmoid(weight)

        # 将维度为b, c的weight, reshape成b, c, 1, 1 与 输入x 相乘
        h, w = weight.shape
        # 通道注意力Mc
        Mc = torch.reshape(weight, (h, w, 1, 1))

        # 乘积获得结果
        x = Mc * x

        return x


class SpatialAttentionModul(nn.Module):  # 空间注意力模块
    def __init__(self, in_channel):
        super(SpatialAttentionModul, self).__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x维度为 [N, C, H, W] 沿着维度C进行操作, 所以dim=1, 结果为[N, H, W]
        MaxPool = torch.max(x, dim=1).values  # torch.max 返回的是索引和value， 要用.values去访问值才行！
        AvgPool = torch.mean(x, dim=1)

        # 增加维度, 变成 [N, 1, H, W]
        MaxPool = torch.unsqueeze(MaxPool, dim=1)
        AvgPool = torch.unsqueeze(AvgPool, dim=1)

        # 维度拼接 [N, 2, H, W]
        x_cat = torch.cat((MaxPool, AvgPool), dim=1)  # 获得特征图

        # 卷积操作得到空间注意力结果
        x_out = self.conv(x_cat)
        Ms = self.sigmoid(x_out)

        # 与原图通道进行乘积
        x = Ms * x

        return x


if __name__ == '__main__':
    inputs = torch.randn(10, 100, 224, 224)
    model = CBAM(in_channel=100)  # CBAM模块, 可以插入CNN及任意网络中, 输入特征图in_channel的维度
    print(model)
    outputs = model(inputs)
    print("输入维度:", inputs.shape)
    print("输出维度:", outputs.shape)


__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}
#_make_divisible: 用于计算某个数的最接近的可以被某个数整除的数。在 MobileNetV2 中用于确定网络层的输出通道数。ConvBNReLU: 封装了一个卷积、BN 和 ReLU6 的组合，用于构建 MobileNetV2 中的基础组件。
# InvertedResidual: MobileNetV2 中的基础单元，采用了轻量级的倒残差结构。MobileNetV2: 整个网络的主体结构，包含了若干个 InvertedResidual 模块，最后接上一个全局平均池化和一个全连接层用于分类。其中的 width_mult 参数用于控制网络宽度。
def _make_divisible(v, divisor, min_value=None):
    # 定义_make_divisible函数
    # 将输入的v除以divisor，得到的结果四舍五入后乘以divisor
    # 然后将这个结果和min_value比较，得到一个新的结果new_v
    # 如果new_v小于0.9*v，则将new_v加上divisor
    if min_value is None: # 如果没有设置最小值，那么最小值默认为除数
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)# 计算新的可被除数，要求其不能小于最小值，并且要被除数整除
    if new_v < 0.9 * v:# 如果新的可被除数比原来的值小 0.9 倍，那么就将新的可被除数加上除数
        new_v += divisor
    return new_v
# 定义ConvBNReLU类，继承自nn.Sequential
# 初始化函数接受in_planes, out_planes, kernel_size, stride, groups五个参数
# 并按照Convolution + BatchNorm + ReLU的顺序依次添加了三个子层
class ConvBNReLU(nn.Sequential):#ConvBNReLU类表示的是卷积、批量归一化和ReLU激活函数的组合
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2# 计算卷积层的 padding
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )
# 定义InvertedResidual类，继承自nn.Module
# 初始化函数接受inp, oup, stride, expand_ratio四个参数
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        # 计算hidden_dim，hidden_dim是InvertedResidual中间层的输出通道数
        hidden_dim = int(round(inp * expand_ratio))
        # 判断是否需要使用shortcut
        # 如果stride为1且输入通道数等于输出通道数，则可以使用shortcut
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # 如果expand_ratio不等于1，添加一个1x1卷积层
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # 添加一个depthwise卷积层
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # 添加一个1x1卷积层
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            # 如果可以使用shortcut，则将输入和输出相加
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None: # 如果没有传入inverted_residual_setting，则按照默认设置定义
            #这是MobileNetV2网络中的inverted residual block设置，共有7个block，每个block的参数含义如下 t: 扩展比例，表示输入特征图的通道数相对于输出特征图的通道数的倍数 c: 输出特征图的通道数
            # n: block中重复的数目
            # s: 步长，表示输出特征图相对于输入特征图的下采样倍数
            # 例如，第一个block的参数为[1, 16, 1, 1]，表示输入的大小为112112x32，扩展比例为1，输出通道数为16，重复一次，步长为1，输出大小为 112x112x16。
            # 最后一个block的参数为[6, 320, 1, 1]，表示输入的大小为7x7x160，扩展比例为6，输出通道数为320，重复一次，步长为1，输出大小为7x7x320。
            inverted_residual_setting = [
                # t, c, n, s
                # 112, 112, 32 -> 112, 112, 16
                [1, 16, 1, 1],
                # 112, 112, 16 -> 56, 56, 24
                [6, 24, 2, 2],
                # 56, 56, 24 -> 28, 28, 32
                [6, 32, 3, 2],
                # 28, 28, 32 -> 14, 14, 64
                [6, 64, 4, 2],
                # 14, 14, 64 -> 14, 14, 96
                [6, 96, 3, 1],
                # 14, 14, 96 -> 7, 7, 160
                [6, 160, 3, 2],
                # 7, 7, 160 -> 7, 7, 320
                [6, 320, 1, 1],
            ]

            # 检查传入的inverted_residual_setting是否符合要求
            if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
                raise ValueError("inverted_residual_setting should be non-empty "
                                 "or a 4-element list, got {}".format(inverted_residual_setting))

            # 计算输入通道数的下一个最小倍数
            input_channel = _make_divisible(input_channel * width_mult, round_nearest)
            # 计算最后一层输出通道数的下一个最小倍数
            self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
            # 添加一个CBAM模块，对输入的3通道特征进行处理
            self.cbam = CBAM(in_channel=3)
            # 将输入的224x224的图像进行卷积，将其变为112x112，同时通道数变为32
            features = [ConvBNReLU(3, input_channel, stride=2)]
            # 遍历所有inverted_residual_setting，依次添加InvertedResidual模块
            for t, c, n, s in inverted_residual_setting:
                # 计算当前层输出通道数的下一个最小倍数
                output_channel = _make_divisible(c * width_mult, round_nearest)
                # 添加n个InvertedResidual模块
                for i in range(n):#循环n次，每次都添加一个block
                    stride = s if i == 0 else 1# 每次循环第一次的步长为s，其他为1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))# 使用给定的block构建一个特征层
                input_channel = output_channel# 更新输入通道数为输出通道数

        # 7, 7, 320 -> 7,7,1280
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1)) # 添加一个ConvBNReLU层，将最后的输出特征层转换为1x1xlast_channel大小
        self.features = nn.Sequential(*features) # 将所有特征层组合为序列，构建特征提取网络

        # 构建分类器，包含一个20%的Dropout层和一个全连接层
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
            #nn.Linear(num_classes*10, num_classes),
        )
        # 对所有模型层进行初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')  # 对卷积层权重进行Kaiming初始化
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # 如果有偏置项，则将其初始化为0
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)  # 对BN层权重初始化为1
                nn.init.zeros_(m.bias)  # 对BN层偏置项初始化为0
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # 对线性层权重进行正态分布初始化
                nn.init.zeros_(m.bias)  # 对线性层偏置项初始化为0

    # 前向传播函数
    def forward(self, x):
        x = self.cbam(x)  # 使用CBAM对输入图像进行特征提取
        x = self.features(x)  # 将输入图像特征进行特征提取
        # 1280
        x = x.mean([2, 3])  # 对特征层进行平均池化，将3维特征层变为1维
        x = self.classifier(x)  # 进行分类预测
        return x
    
    def freeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def Unfreeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = True


def mobilenet_v2(pretrained=False, progress=True, num_classes=1000):
    model = MobileNetV2()
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'], model_dir='./model_data',
                                              progress=progress)
        model.load_state_dict(state_dict)

    if num_classes != 1000:
        model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(model.last_channel, num_classes),
            )
    return model


if __name__ == "__main__":
    model = mobilenet_v2()
    print(model)
