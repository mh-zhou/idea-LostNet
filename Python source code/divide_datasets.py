import os
import shutil
import random
path = './datasets/train'
dst = './datasets/test'
fire_list = os.listdir(path)
print(fire_list)
for i in fire_list:
    print(i)
    img_path = os.path.join(path, i)
    img_dst = os.path.join(dst, i)
    if not os.path.exists(img_dst):
        os.makedirs(img_dst)
    img_list = os.listdir(img_path)
    random.shuffle(img_list)
    # num = 0
    # for j in img_list:
    #     print(j)
    #     os.rename(os.path.join(img_path, j), os.path.join(img_path, str(num).rjust(4, '0')+'.jpg'))
    #     num += 1
    for j in range(int(len(img_list)*0.2)):
        shutil.move(os.path.join(img_path, img_list[j]), os.path.join(img_dst, img_list[j]))