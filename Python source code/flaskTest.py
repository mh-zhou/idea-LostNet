# utf-8
from flask import Flask, request
import warnings
import torch
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms, models
from mobilenet import MobileNetV2

app = Flask(__name__)
warnings.filterwarnings("ignore")
# classes = ['USBflashdisk','bag','book','card','earphone','key','lipstick','phone','umbrella','vacuumcup',]    #标签序号对应类名
classes = ['USBflashdisk','bag','book','card','earphone','key','lipstick','phone','umbrella','vacuumcup',]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 是否使用gpu加速
num_classes = 10



def test_mydata(a):  # 定义预测函数
    # 调整图片大小
    im = plt.imread(a)  # 读入图片
    images = Image.open(a)  # 将图片存储到images里面
    images = images.resize((224, 224))  # 调整图片的大小为224*224的大小
    images = images.convert('RGB')  # RGB化

    transform = transforms.ToTensor()
    images = transform(images)  # 图像转化成tensor类型
    images = images.resize(1, 3, 224, 224)  # 调整输入网络图片大小
    images = images.to(device)  # gpu加速

    path_model = "./96.8cbam.pth"  # 调用训练好的模型
    model = torch.load(path_model)
    model = model.to(device)

    model.eval()
    outputs = model(images)  # 将图片传入模型预测
    values, indices = outputs.data.max(1)  # 返回最大概率值和下标  output不是tensor类型所以要加.data
    result = classes[int(indices[0])]
    print(result)
    return result


def Diver(url):
    #url=r"https://guli-0041.oss-cn-beijing.aliyuncs.com/avatar/2021/11/14/b65cf7e4-73f8-4e46-aefc-4894ce5105ab.jpg"
    # img = input('Input image filename:')
    #img = "E:\mache\img\phone\phone01.jpg"
    # try:
    #     image = Image.open(url)
    # except:
    #     print('Open Error! Try again!')
    # else:
    r_image = test_mydata(url)
    return r_image


@app.route('/')
def index():
    url = request.args.get('url')
    result = Diver(url)
    return {'result':result}


if __name__ == '__main__':
    app.debug = True  # 设置调试模式，生产模式的时候要关掉debug
    app.run()
