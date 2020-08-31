import tensorflow as tf
import numpy as np
from collections import Counter
import scipy.io as sio
import os
import matplotlib.pyplot as plt
import pathlib
import math
class Model():

    def __init__(self,args,sess):
        self.sess = sess
        self.args = args
        self.result = args.result
        info = sio.loadmat(os.path.join(self.result,'info.mat'))
        self.shape = info['shape']
        self.dim = info['dim']
        self.class_num = int(info['class_num'])
        self.data_gt = info['data_gt']
        self.log = args.log
        self.model = args.model
        self.data_name = args.data_name
        self.cube_size = args.cube_size
        self.data_path = args.data_path
        self.epoch = args.epoch
        self.tfrecords = args.tfrecords
        self.global_step = tf.Variable(0,trainable=False)
        if args.use_lr_decay: #decays every  decay_steps by multiplying decay_rate
            self.lr = tf.train.exponential_decay(learning_rate=args.lr,
                                             global_step=self.global_step,
                                             decay_rate=args.decay_rate,
                                             decay_steps=args.decay_steps)
        else:
            self.lr = args.lr
        # feature map to input network
        self.image = tf.placeholder(dtype=tf.float32, shape=(None, self.cube_size,self.cube_size,self.dim))
        # Corresponding labels
        self.label = tf.placeholder(dtype=tf.int64, shape=(None, 1))
        self.classifer = self.classifer

        self.pre_label = self.classifer(self.image)
        self.model_name = os.path.join('model.ckpt')
        self.loss()
        self.summary_write = tf.summary.FileWriter(os.path.join(self.log),graph=self.sess.graph)
        self.saver = tf.train.Saver(max_to_keep=100)

    def loss(self):
        with tf.variable_scope('loss'): # Calculate the loss function and optimize network parameters
            loss_cross_entropy = tf.losses.sparse_softmax_cross_entropy(self.label,self.pre_label,scope='loss_cross_entropy')
            loss_cross_entropy = tf.reduce_mean(loss_cross_entropy)
            self.loss_total = loss_cross_entropy
            tf.summary.scalar('loss_total',self.loss_total)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss_total,global_step=self.global_step)
        self.merged = tf.summary.merge_all()


    def Cal_net_param_2D(self,feature,col,row,f_num):
        '''
        :param feature: feature map
        :param col:  col of feature map
        :param row:  row of eature map
        :param f_num: kernel size
        :return: Returns the number of parameters(2D-CNN)
        '''
        shape = int(feature.get_shape().as_list()[3])
        param = shape * col * row + 1
        param = param * f_num
        return param

    def Cal_net_param_3D(self,feature,col,row,width,f_num):
        '''

        :param feature: feature map
        :param col: col of feature map
        :param row: row of eature map
        :param width: width of eature map
        :param f_num: kernel size
        :return: Returns the number of parameters(3D-CNN)
        '''
        shape = int(feature.get_shape().as_list()[4])
        param = shape * col * row * width + 1
        param = param * f_num
        return param

    def NonLocalBlock(self, input, subsample=False):
        '''
        @Non-local Neural Networks
        Non-local Block
        :param input: feature map
        :param subsample:
        :return: correlation feature maps, parameters  used in NonLocalBlock
        '''
        param_total = 0
        _, height, width, channel = input.get_shape().as_list()  # (B, H, W, C)

        theta = tf.layers.conv2d(input, channel // 2, 1)  # (B, H, W, C // 2)
        param = self.Cal_net_param_2D(input, 1, 1, channel//2)
        param_total += param
        theta = tf.reshape(theta, [-1, height * width, channel // 2])  # (B, H*W, C // 2)

        phi = tf.layers.conv2d(input, channel // 2, 1)  # (B, H, W, C // 2)
        param = self.Cal_net_param_2D(input, 1, 1, channel // 2)
        param_total += param
        if subsample:
            phi = tf.layers.max_pooling2d(phi, 2, 2)  # (B, H / 2, W / 2, C // 2)
            phi = tf.reshape(phi, [-1, height * width // 4, channel // 2])  # (B, H * W / 4, C // 2)
        else:
            phi = tf.reshape(phi, [-1, height * width, channel // 2])  # (B, H*W, C // 2)
        phi = tf.transpose(phi, [0, 2, 1])  # (B, C // 2, H*W)

        f = tf.matmul(theta, phi)  # (B, H*W, H*W)
        f = tf.nn.softmax(f)  # (B, H*W, H*W)

        g = tf.layers.conv2d(input, channel // 2, 1)  # (B, H, W, C // 2)
        param = self.Cal_net_param_2D(input, 1, 1, channel // 2)
        param_total += param
        if subsample:
            g = tf.layers.max_pooling2d(g, 2, 2)  # (B, H / 2, W / 2, C // 2)
            g = tf.reshape(g, [-1, height * width // 4, channel // 2])  # (B, H*W, C // 2)
        else:
            g = tf.reshape(g, [-1, height * width, channel // 2])  # (B, H*W, C // 2)

        y = tf.matmul(f, g)  # (B, H*W, C // 2)
        y = tf.reshape(y, [-1, height, width, channel // 2])  # (B, H, W, C // 2)
        param = self.Cal_net_param_2D(y, 1, 1, channel)
        param_total += param
        y = tf.layers.conv2d(y, channel, 1)  # (B, H, W, C)


        y = tf.add(input, y)  # (B, W, H, C)

        return y, param_total

    def ConvolutionBlock(self, conv, f_num):
        '''
        :param conv:  feature map
        :param f_num: kernel size
        :return: new feature map ,parameters  used in ConvolutionBlock
        '''
        param_total = 0
        conv, param= self.NonLocalBlock(conv)
        param_total += param
        conv0 = tf.layers.conv2d(conv, f_num, (3, 3), strides=(1, 1), padding='same',
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                  bias_initializer=tf.constant_initializer(0))
        conv0 = tf.layers.batch_normalization(conv0)
        conv0 = tf.nn.relu(conv0)
        param = self.Cal_net_param_2D(conv, 3, 3, f_num)
        param_total = param_total + param
        conv10 = conv + conv0
        conv1 = tf.layers.conv2d(conv10, f_num, (3, 3), strides=(1, 1), padding='same',
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                  bias_initializer=tf.constant_initializer(0))
        conv1 = tf.layers.batch_normalization(conv1)
        conv1 = tf.nn.relu(conv1)
        param = self.Cal_net_param_2D(conv10, 3, 3, f_num)
        param_total = param_total + param
        conv11 = conv + conv0 + conv1
        conv2 = tf.layers.conv2d(conv11, f_num, (3, 3), strides=(1, 1), padding='same',
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                  bias_initializer=tf.constant_initializer(0))
        conv2 = tf.layers.batch_normalization(conv2)
        conv2 = tf.nn.relu(conv2)
        param = self.Cal_net_param_2D(conv11, 3, 3, f_num)
        param_total = param_total + param
        conv12 =conv2 + conv1
        conv3 = tf.layers.conv2d(conv12, f_num, (3, 3), strides=(1, 1), padding='same',
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                  bias_initializer=tf.constant_initializer(0))
        conv3 = tf.layers.batch_normalization(conv3)
        conv3 = tf.nn.relu(conv3)
        param = self.Cal_net_param_2D(conv12, 3, 3, f_num)
        param_total = param_total + param
        conv13 = conv3 + conv2
        conv4 = tf.layers.conv2d(conv13, f_num, (3, 3), strides=(1, 1), padding='same',
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                  bias_initializer=tf.constant_initializer(0))
        conv4 = tf.layers.batch_normalization(conv4)
        conv4 = tf.nn.relu(conv4)
        param = self.Cal_net_param_2D(conv13, 3, 3, f_num)
        param_total = param_total + param
        conv5 = conv4 + conv3 + conv
        return conv5, param_total

    def Convolution3_D(self,feature,f_num):# 3D-CNN
        param_total = 0
        Flops=0
        with tf.variable_scope('conv0'):
            conv0 = tf.layers.conv3d(feature, f_num, (3, 3, 4), strides=(1, 1, 1), padding='valid',
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                     bias_initializer=tf.constant_initializer(0))
            #conv0 = tf.layers.batch_normalization(conv0)
            conv0 = tf.nn.relu(conv0)
            param = self.Cal_net_param_3D(feature, 3, 3, 4, f_num)
            param_total = param_total + param
            Flops+= param*36
            #print(conv0, 'Param:', param)
        with tf.variable_scope('conv1'):
            conv1 = tf.layers.conv3d(conv0, f_num * 2, (3, 3, 4), strides=(1, 1, 1), padding='valid',
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                     bias_initializer=tf.constant_initializer(0))
            #conv1 = tf.layers.batch_normalization(conv1)
            conv1 = tf.nn.relu(conv1)
            param = self.Cal_net_param_3D(conv0, 3, 3, 4, f_num*2)
            param_total = param_total + param
            Flops += param * 36
        with tf.variable_scope('conv2'):
            conv2 = tf.layers.conv3d(conv1, f_num * 4, (2, 2, 2), strides=(2, 2, 2), padding='valid',
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                     bias_initializer=tf.constant_initializer(0))
            #conv2 = tf.layers.batch_normalization(conv2)
            conv2 = tf.nn.relu(conv2)
            param = self.Cal_net_param_3D(conv1, 2, 2, 2, f_num * 4)
            param_total = param_total + param
            Flops += param * 8
        # with tf.variable_scope('pool'):
        #     # pool0 = tf.layers.max_pooling3d(conv2,(2,2,2),strides=(2, 2, 1),padding='valid')
        #     # conv6 = pool0
            conv6 = tf.layers.conv3d(conv2, f_num * 8, (3, 3, 2), strides=(1, 1, 1), padding='valid',
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                     bias_initializer=tf.constant_initializer(0))
            conv6 = tf.layers.batch_normalization(conv6)
            conv6 = tf.nn.relu(conv6)
            param = self.Cal_net_param_3D(conv2, 3, 3, 2, f_num * 8)
            param_total = param_total + param
            Flops += param * 18
            # print(conv1)
        with tf.variable_scope('reshape'):
            shape = conv0.get_shape().as_list()
            conv3 = tf.reshape(conv0, [-1, shape[1], shape[2], shape[3] * shape[4]])
            shape = conv1.get_shape().as_list()
            conv4 = tf.reshape(conv1, [-1, shape[1], shape[2], shape[3] * shape[4]])
            shape = conv2.get_shape().as_list()
            conv5 = tf.reshape(conv2, [-1, shape[1], shape[2], shape[3] * shape[4]])
            shape = conv6.get_shape().as_list()
            conv7 = tf.reshape(conv6, [-1, shape[1], shape[2], shape[3] * shape[4]])
            return conv3, conv4 ,conv5, conv7, param_total,Flops

    def Convolution2_D(self, feature, f_num): # 2D-CNN
        param_total = 0
        Flops=0
        with tf.variable_scope('conv2'):
            conv0 = tf.layers.conv2d(feature, f_num , (3, 3), strides=(1, 1), padding='valid',
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                      bias_initializer=tf.constant_initializer(0))
            #conv0 = tf.layers.batch_normalization(conv0)
            conv0 = tf.nn.relu(conv0)
            param = self.Cal_net_param_2D(feature, 3, 3, f_num)
            param_total = param_total + param
            Flops+= param*9
            #print(conv0)
        # with tf.variable_scope('block0'):
        #     conv1 = self.ConvolutionBlock(conv0,f_num )
        with tf.variable_scope('conv3'):
            conv2 = tf.layers.conv2d(conv0, f_num * 2, (3, 3), strides=(1, 1), padding='valid',
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                      bias_initializer=tf.constant_initializer(0))
            #conv2 = tf.layers.batch_normalization(conv2)
            conv2 = tf.nn.relu(conv2)
            param = self.Cal_net_param_2D(conv0, 3, 3, f_num*2)
            param_total = param_total + param
            Flops += param * 9
            #print(conv2)

        with tf.variable_scope('conv4'):
            conv4 = tf.layers.conv2d(conv2, f_num * 4, (2, 2), strides=(2, 2), padding='valid',
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                      bias_initializer=tf.constant_initializer(0))
            #conv4 = tf.layers.batch_normalization(conv4)
            conv4 = tf.nn.relu(conv4)
            param = self.Cal_net_param_2D(conv2, 2, 2, f_num * 4)
            param_total = param_total + param
            Flops += param * 4

        with tf.variable_scope('conv6'):
            conv6 = tf.layers.conv2d(conv4, f_num * 4, (3, 3), strides=(1, 1), padding='valid',
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                      bias_initializer=tf.constant_initializer(0))
            conv6= tf.layers.batch_normalization(conv6)
            conv6 = tf.nn.relu(conv6)
            param = self.Cal_net_param_2D(conv4, 3, 3, f_num * 4)
            param_total = param_total + param
            Flops += param * 9
        return conv0, conv2, conv4 ,conv6 ,param_total,Flops

    def classifer(self, feature2D): # classifer, which represents the entire network framework
        f_num = 8
        param_total = 0
        Flops=0
        feature3D = tf.expand_dims(feature2D, 4)
        print(feature2D)
        with tf.variable_scope('Convolution3D'): #3D-CNN
            conv0,conv1,conv2 ,conv00,param_total_3D,Flops_3D= self.Convolution3_D(feature3D, f_num)
            param_total = param_total + param_total_3D
            Flops+= Flops_3D
        with tf.variable_scope('Convolution2D'):#2D-CNN
            conv3,conv4,conv5, conv01,param_total_2D,Flops_2D= self.Convolution2_D(feature2D, f_num*2)
            param_total = param_total + param_total_2D
            Flops += Flops_2D
        with tf.variable_scope('concat'):
            conv6 = tf.concat([conv0, conv3],axis=3)
            shape_conv6 = conv6.get_shape().as_list()[3]
            conv6 ,param= self.ConvolutionBlock(conv6, shape_conv6)
            param_total = param_total + param
            Flops += param*9
            conv7 = tf.concat([conv1, conv4],axis=3)
            shape = conv7.get_shape().as_list()[3]
            conv7 ,param = self.ConvolutionBlock(conv7, shape)
            param_total = param_total + param
            Flops += param * 9
            conv8 = tf.concat([conv2, conv5], axis=3)
            shape = conv8.get_shape().as_list()[3]
            conv8 ,param= self.ConvolutionBlock(conv8, shape)
            param_total = param_total + param
            Flops += param * 9
        with tf.variable_scope('first_route'):
            pool0 = tf.layers.max_pooling2d(conv6, (2, 2), strides=(2, 2), padding='valid')
            conv62 = tf.layers.conv2d(pool0, shape_conv6,(2, 2), strides=(1, 1), padding='valid',
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                     bias_initializer=tf.constant_initializer(0))
            conv62 = tf.layers.batch_normalization(conv62)
            conv62 = tf.nn.relu(conv62)
            param = self.Cal_net_param_2D(pool0, 2, 2, shape_conv6)
            param_total = param_total + param
            Flops += param * 4
            print(conv62)
            pool1 = tf.layers.max_pooling2d(conv7, (2, 2), strides=(2, 2), padding='valid')
            print(pool1)
            conv9 = tf.concat([conv62 ,pool1,conv8],axis=3)
            pool2 = conv9
            print(pool2)
        with tf.variable_scope('conv5'):
            shape0 = pool2.get_shape().as_list()[1]
            shape1 = pool2.get_shape().as_list()[3]
            conv10 = tf.layers.conv2d(pool2, shape1, (shape0, shape0), strides=(1, 1), padding='valid',
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                     bias_initializer=tf.constant_initializer(0))
            param = self.Cal_net_param_2D(pool2, shape0, shape0, shape1)
            param_total = param_total + param
            Flops += param * shape0 * shape0
            conv10 = tf.concat([conv10,conv00,conv01],axis=3)
            print(conv10)
        with tf.variable_scope('flatten'):
            feature = tf.layers.conv2d(conv10, self.class_num, (1, 1), strides=(1, 1), padding='valid',
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                     bias_initializer=tf.constant_initializer(0))
            feature = tf.layers.flatten(feature)
            param = self.Cal_net_param_2D(conv10, 1, 1, self.class_num)
            param_total = param_total + param
            Flops += param
            print(feature, 'Param:', param)
            print('Param_total_number:',param_total)
            print('Flops:', Flops)
        return feature

    def load(self, checkpoint_dir):# Loading model
        print("Loading model ...")
        model_name = os.path.join(checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(model_name)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(model_name, ckpt_name))
            print("Load successful.")
            return True
        else:
            print("Load fail!!!")
            exit(0)

    def train_prelabel(self,train_pre_label,train_label):# Calculate training set accuracy
        '''

        :param train_pre_label: the predict label of train set
        :param train_label: training labels
        :return: the accuracy of training set
        '''
        train_pre_label = np.argmax(train_pre_label, 1)
        train_pre_label = np.expand_dims(train_pre_label, 1)
        train_accuracy = np.sum((train_pre_label == train_label))/train_label.shape[0]
        return train_accuracy

    def loss_store_acc(self,trainloss,trainaccuracy):# Store training losses and training accuracy
        if not os.path.exists(str(pathlib.Path(self.data_name))):
            os.mkdir(str(pathlib.Path(self.data_name)))
        sio.savemat(os.path.join(str(pathlib.Path(self.data_name)), 'loss'+ '.mat'),
                    {'loss': trainloss})
        sio.savemat(os.path.join(str(pathlib.Path(self.data_name)), 'accuracy'  + '.mat'),
                    {'accuracy': trainaccuracy})

    def train(self,train_dataset,dataset):#train
        '''
        :param train_dataset: train set
        :param dataset: dataser
        :return: no return
        '''
        init = tf.global_variables_initializer()
        self.sess.run(init)
        trainloss=[]
        trainaccuracy=[]
        for i in range(self.epoch):
            train_data, train_label = self.sess.run(train_dataset)
            # print(train_data.shape,train_label.shape)
            l, _, summary, lr ,train_pre_label= self.sess.run([self.loss_total, self.optimizer, self.merged, self.lr ,self.pre_label],
                                              feed_dict={self.image: train_data, self.label: train_label})
            train_accuracy = self.train_prelabel(train_pre_label,train_label)
            trainloss.append(l)
            trainaccuracy.append(train_accuracy)
            #self.loss_store_acc(trainloss,trainaccuracy) #procucting loss and training_accuracy
            if i % 1000 == 0:
                print(i, 'step:', l, 'learning rate:', lr,'train_accuracy:',train_accuracy)
            if i % (self.epoch-1) == 0 and i>0:
                self.saver.save(self.sess, os.path.join(self.model, self.model_name), global_step=i)
                print('saved...')
                self.test(dataset)
            self.summary_write.add_summary(summary, i)


    def test(self,dataset):#test
        test_dataset = dataset.data_parse(os.path.join(self.tfrecords, 'test_data.tfrecords'), type='test')
        acc_num,test_num = 0,0
        matrix = np.zeros((self.class_num,self.class_num),dtype=np.int64)
        result_label= []
        try:
            while True:
                test_data, test_label = self.sess.run(test_dataset)
                # pre_label: producting labels, test_data:test data ,test_label: the labels of test data
                pre_label = self.sess.run(self.pre_label, feed_dict={self.image:test_data,self.label:test_label})
                pre_label = np.argmax(pre_label,1)
                pre_label = np.expand_dims(pre_label,1)
                result_label.append(pre_label)
                acc_num += np.sum((pre_label==test_label))
                test_num += test_label.shape[0]
                print(acc_num,test_num,acc_num/test_num)
                for i in range(pre_label.shape[0]):
                    matrix[pre_label[i],test_label[i]]+=1
        except tf.errors.OutOfRangeError:
            print("test end!")

        ac_list = []
        for i in range(len(matrix)): # Confusion matrix
            ac = matrix[i, i] / sum(matrix[:, i])
            ac_list.append(ac)
            print(i+1,'class:','(', matrix[i, i], '/', sum(matrix[:, i]), ')', ac)
        print('confusion matrix:')
        print(np.int_(matrix))
        print('total right num:', np.sum(np.trace(matrix)))
        accuracy = np.sum(np.trace(matrix)) / np.sum(matrix)
        print('oa:', accuracy)
        # kappa
        kk = 0
        for i in range(matrix.shape[0]):# calculate kappa
            kk += np.sum(matrix[i]) * np.sum(matrix[:, i])
        pe = kk / (np.sum(matrix) * np.sum(matrix))
        pa = np.trace(matrix) / np.sum(matrix)
        kappa = (pa - pe) / (1 - pe)
        ac_list = np.asarray(ac_list)
        aa = np.mean(ac_list)
        oa = accuracy
        print('aa:',aa)
        print('kappa:', kappa)
        return oa, aa, kappa , matrix ,result_label

    def all_data(self,dataset):# Produce all forecast labels including background
        all_dataset = dataset.data_parse(os.path.join(self.tfrecords, 'all_data.tfrecords'), type='all')
        result_label = []
        try:
            while True:
                all_data= self.sess.run(all_dataset)
                pre_label = self.sess.run(self.pre_label, feed_dict={self.image: all_data})
                pre_label = np.argmax(pre_label, 1)
                pre_label = np.expand_dims(pre_label, 1)
                result_label.append(pre_label)
        except tf.errors.OutOfRangeError:
            print("All end!")
        return result_label
