import argparse
import os
import tensorflow as tf
import numpy as np
import scipy.io as sio
from data_loader import Data
from model import Model
import label2color
import time,pathlib

parser = argparse.ArgumentParser(description='Copresion and Classification for HSI')

parser.add_argument('--result',dest='result',default='result')
parser.add_argument('--log',dest='log',default='log')
parser.add_argument('--model',dest='model',default='model')
parser.add_argument('--tfrecords',dest='tfrecords',default='tfrecords')#Store parameters in the network
parser.add_argument('--data_name',dest='data_name',default='PaviaU')# the name of dataset
parser.add_argument('--data_path',dest='data_path',default="E:\Datas")# The local address where datasets are stored

parser.add_argument('--use_lr_decay',dest='use_lr_decay',default=True)
parser.add_argument('--padding',dest='padding',default=5 )
parser.add_argument('--decay_rate',dest='decay_rate',default=0.99)
parser.add_argument('--learning_rate',dest='lr',default=0.001) #Learning rate
parser.add_argument('--train_num',dest='train_num',default=0.1) # intger for number and decimal for percentage
parser.add_argument('--batch_size',dest='batch_size',default=80)# batch_size of training
parser.add_argument('--fix_seed',dest='fix_seed',default=False)
parser.add_argument('--seed',dest='seed',default=666)
parser.add_argument('--decay_steps',dest='decay_steps',default=5000)
parser.add_argument('--test_batch',dest='test_batch',default=2000)
parser.add_argument('--epoch',dest='epoch',default=6001) # Default training times
parser.add_argument('--n_components',dest='n_components',default=10) # Principal component size
parser.add_argument('--cube_size',dest='cube_size',default=11) #cube_size=2*padding+1
parser.add_argument('--load_model',dest='load_model',default=False)

#Create few files to hold the generated data
args = parser.parse_args()
if not os.path.exists(args.model):
    os.mkdir(args.model)
if not os.path.exists(args.log):
    os.mkdir(args.log)
if not os.path.exists(args.result):
    os.mkdir(args.result)
if not os.path.exists(args.tfrecords):
    os.mkdir(args.tfrecords)

#Used to draw a tag diagram with no background
def decode_map(test_pos_all,pre_label,data_name):
    '''
    :param test_pos_all: The location of label
    :param pre_label: Predicting labels
    :param data_name: The name of data
    :return: no return
    '''
    info = sio.loadmat(os.path.join(args.result, 'info.mat'))
    data_gt = info['data_gt']
    k = 0
    pre_label = np.array(pre_label)
    label = pre_label[0]
    print(label.shape)
    if label.shape[1] == 1:
        for i in range(1,pre_label.shape[0]):
            label = np.vstack((label,pre_label[i]))
    else:
        for i in range(1,pre_label.shape[0]):
            label = np.hstack((label,pre_label[i]))
    pre_label =  np.reshape(label, [-1])
    for i in test_pos_all:
        [r, c] = i[1]
        data_gt[r-args.padding, c-args.padding] = pre_label[k]+1
        k += 1
    label2color.draw_RGB(data_gt, data_name,0)
    #label2color.False3color(data_name) #Used to draw false color diagrams

#Used to draw a tag diagram with a background
def create_All_label(model,data_name,dataset,num):
    '''
    :param model: Trained model
    :param data_name: The name of data
    :param dataset: Dataset
    :param num: Number of runs
    :return: no return
    '''
    info = sio.loadmat(os.path.join(args.result, 'info.mat'))
    data = info['data']
    print(data.shape)
    all_data_name = os.path.join(args.tfrecords, 'all_data.tfrecords')
    writer = tf.python_io.TFRecordWriter(all_data_name)
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    pixel_t=None
    for i in range(args.padding,data.shape[0]-args.padding):
        for j in range(args.padding, data.shape[1] - args.padding):
            pixel_t = data[i-args.padding:i+args.padding+1,j-args.padding:j+args.padding+1,:]
            pixel_t = pixel_t.astype(np.float32).tostring()
            example = tf.train.Example(features=(tf.train.Features(
                feature={
                    'alldata': _bytes_feature(pixel_t)
                }
            )))
            writer.write(example.SerializeToString())
    writer.close()
    #all_dataset = dataset.data_parse(os.path.join(args.tfrecords, 'all_data.tfrecords'), type='all')
    model.load(args.model)
    pre_label= model.all_data(dataset)
    pre_label = np.array(pre_label)
    label = pre_label[0]
    print(label.shape)
    if label.shape[1] == 1:
        for i in range(1, pre_label.shape[0]):
            label = np.vstack((label, pre_label[i]))
    else:
        for i in range(1, pre_label.shape[0]):
            label = np.hstack((label, pre_label[i]))
    pre_label = np.reshape(label, [-1])
    print(pre_label.shape)
    label= np.reshape(pre_label,[data.shape[0] - 2*args.padding,data.shape[1] - 2*args.padding])
    label2color.draw_RGB(label+1, data_name,num)

def main(num,run_num):

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    overall_oa=[] # used to store OA
    overall_aa = [] # used to store AA
    overall_kappa = [] # used to store Kappa
    overall_matrix=[] # used to Used to store the resulting obfuscation matrix
    for j in range(num): # In this experiment, num is 20
        print('第'+str(j+1)+"次运行：")
        for i in range(1):
            args.id = str(i)
            tf.reset_default_graph()
            with tf.Session(config=config) as sess:
                args.result = os.path.join(args.result,args.id)
                args.log = os.path.join(args.log, args.id)
                args.model = os.path.join(args.model, args.id)
                args.tfrecords = os.path.join(args.tfrecords, args.id)
                if not os.path.exists(args.model):
                    os.mkdir(args.model)
                if not os.path.exists(args.log):
                    os.mkdir(args.log)
                if not os.path.exists(args.result):
                    os.mkdir(args.result)
                if not os.path.exists(args.tfrecords):
                    os.mkdir(args.tfrecords)

                dataset = Data(args)
                #test_pos_all is used to draw a diagram with no background
                test_pos_all = dataset.read_data()
                # producing training dataset
                train_dataset = dataset.data_parse(os.path.join(args.tfrecords, 'train_data.tfrecords'), type='train')
                # producing testing dataset, or producing during use.
                test_dataset = dataset.data_parse(os.path.join(args.tfrecords, 'test_data.tfrecords'), type='test')
                #all_dataset = dataset.data_parse(os.path.join(args.tfrecords, 'map_data.tfrecords'), type='map')

                model = Model(args,sess)
                # traing the model
                if not args.load_model:
                    model.train(train_dataset,dataset)
                else:
                    model.load(args.model)
                # the model return the result of experiments once.
                oa, aa, kappa, matrix, result_label = model.test(dataset)
                # Calculated running time
                over = time.time()
                print("平均运行时间:  " + str((over - start) / 60.0) + " min")
                #create_All_label(model,args.data_name,dataset,j) # draw picture containing background
                #decode_map(test_pos_all, result_label,args.data_name) # draw picture not containing background
                overall_oa.append(oa)
                overall_aa.append(aa)
                overall_kappa.append(kappa)
                overall_matrix.append(matrix)
                args.result = 'result'
                args.log = 'log'
                args.tfrecords = 'tfrecords'
                args.model = 'model'
    # Store the results of the experiment locally
    if not os.path.exists(str(pathlib.Path(args.data_name))):
        os.mkdir(str(pathlib.Path(args.data_name)))
    sio.savemat(os.path.join(str(pathlib.Path(args.data_name)), 'result' + str(run_num) + '.mat'),
            {'oa': overall_oa, 'aa': overall_aa, 'kappa': overall_kappa})
    sio.savemat(os.path.join(str(pathlib.Path(args.data_name)), 'matrix' + str(run_num) + '.mat'),
            {'matrix': overall_matrix})



if __name__ == '__main__':
    data_name = {0: 'Indian_pines', 1: 'PaviaU', 2: 'Salinas',3:'Houston'}# the name of data
    sample_num = {0: 200, 1: 0.01, 2: 0.03, 3: 0.05, 4: 0.08, 5: 0.1}# intger for number and decimal for percentage
    epoch ={0:2001, 1: 8001, 2: 12001,3: 10001}# The training times are determined according to the loss function
    for i in range(len(data_name)):#len(data_name)
        num = 20
        args.data_name = data_name[i]
        args.epoch = epoch[i]
        print(args.data_name)
        for j in range(1,len(sample_num)):
            args.train_num = sample_num[j]
            start = time.time()
            main(num, j)
            over = time.time()
            average_time = (over - start) / (60.0 * num)
            print("平均运行时间:  " + str(average_time) + " min")
            sio.savemat(os.path.join(str(pathlib.Path(args.data_name)), 'average_time' + str(j) + '.mat'),
                        {'average_time': average_time})