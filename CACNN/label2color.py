import argparse
import pathlib,random
import os
import unit
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Copresion and Classification for HSI')
parser.add_argument('--result',dest='result',default='G:/Second-experiment/Indian Pines/result/0')
args = parser.parse_args()
# from data_loader import Data
# from model import Model
# data_name = 'Houston'
# data_gt_dict = sio.loadmat(str(pathlib.Path('E:\Datas', data_name+'_gt.mat')))
# data_gt_name = [t for t in list(data_gt_dict.keys()) if not t.startswith('__')][0]
# data_gt = data_gt_dict[data_gt_name].astype(np.int64)
def draw_RGB(label,data_name,num):
    '''
    :param label: The producting labels
    :param data_name: The name of data
    :param num: Number of runs
    :return:
    '''
    [w, h] = label.shape
    im = np.zeros(shape=[w, h, 3], dtype=int)
    if data_name == 'Indian_pines':
        map = [0,0,0,  0,0,255,  255,109,0,  255,127,255, 31,127,213,  0,255,0,  181,85,181, 127,127,255, 0,95,159, 128,73,85]

        # map = [140 ,67, 46,0 ,0, 255,255 ,100 ,0,0 ,255 ,123,164 ,75 ,155,101 ,174 ,255,118 ,254 ,172, 60 ,91,
        #        112,255, 255, 0,255, 255 ,125,255, 0 ,255,100 ,0 ,255,0 ,172 ,254, 0 ,255 ,0,171 ,175 ,80,101,
        #        193, 60]
    elif data_name == 'PaviaU':
        map = [0, 0, 0, 192, 192, 192, 0, 255, 0, 0, 255, 255, 0, 128, 0, 255, 0, 255, 165, 82, 41, 128, 0, 128, 255, 0,
               0, 255, 255, 0]
    elif data_name == 'Salinas':
        map = [0,0,0,   0,0,255,  255,100,0,  0,255,134,  150,70,150,   100,150,255,   60,90,114,  255,255,125,
               255,0,255,  100,0,255,  1,170,255,  0,255,0,  175,175,82,   100,190,56,   140,67,46, 115,255,172,
               255,255,0]
    elif data_name == 'Houston':
        map = [0,0,0, 1,208,0,  136,255,0, 49,156,96,  0,143,0,  1,76,0,  255,255,255,  255,0,0,  204,204,204,
               176,0,0,  255,255,0,  255,0,255,   179,204,230]
    else:
        map = [0, 0, 0, 0, 0, 255, 0, 128, 0, 0, 255, 0, 255, 0, 0, 142, 71, 2, 192, 192, 192, 0, 255, 255, 246, 110, 0, 255,
               255, 0]
    map = np.array(map)
    map = np.reshape(map, [-1,3])
    for i in range(w):
        for j in range(h):
            im[i,j] = map[label[i,j],:]
    plt.imshow(im)
    plt.axis('off')
    if not os.path.exists(str(pathlib.Path(data_name))):
        os.mkdir(str(pathlib.Path(data_name)))
    plt.savefig(os.path.join(str(pathlib.Path(data_name)), data_name + str(num) + '.png'), bbox_inches='tight', dpi=700)
    plt.show()

#draw_RGB(data_gt,data_name)

def False3color(data_name):
    # Draw a Falsecolor image based on the dataset name
    if data_name == 'Indian_pines':
        data_dict = sio.loadmat(str(pathlib.Path('E:\Datas', data_name + '.mat')))
        band = [50, 23, 13]
    elif data_name == 'Salinas':
        data_dict = sio.loadmat(str(pathlib.Path('E:\Datas', data_name + '_corrected.mat')))
        band = [50,28,15]
    elif data_name == 'Houston':
        data_dict = sio.loadmat(str(pathlib.Path('E:\Datas', data_name + '.mat')))
        band = [50, 23, 13]
    else:
        data_dict = sio.loadmat(str(pathlib.Path('E:\Datas', data_name + '.mat')))
        band =[60,40,9]
    data_name = [t for t in list(data_dict.keys()) if not t.startswith('__')][0]
    data = data_dict[data_name]
    band = np.array(band, dtype=int)
    data = data[: , : , band]
    data = (data-np.min(data))/(np.max(data)-np.min(data)).astype(np.float32)
    data = np.array(data*255, dtype=int)
    plt.imshow(data)
    plt.axis('off')
    plt.savefig('Three_band_'+data_name + '.png', bbox_inches='tight', dpi=700)
    plt.show()
#False3color(data_name)
#draw_RGB(data_gt,args,data_name)
#False3color(data_name)
