import os
import scipy.io as sio
import numpy as np

run_num = 0
data_name = 'Indian_pines'
path = os.path.join(os.getcwd(),data_name)
path_data = os.path.join(path,'result' + str(run_num)+'.mat')
path__matrix =os.path.join(path,'matrix' +  str(run_num) +'.mat')
path_avg_time = os.path.join(path,'average_time'+str(run_num)+'.mat')
data=sio.loadmat(path_data)
matrix_data=sio.loadmat(path__matrix )
matrix=matrix_data['matrix']
average_time = sio.loadmat(path_avg_time)
average_time = average_time['average_time']
matrix_zero=np.zeros(matrix[0].shape)
oa=data['oa']
aa=data['aa']
kappa=data['kappa']
print(oa)
#length=len(oa)
# path_IN = 'E:\Datas\Layer-num-conv\\train'
# path_data_ = os.path.join(os.path.join(path_IN,data_name),'result' + str(run_num)+'.mat')
# path__matrix_ =os.path.join(os.path.join(path_IN,data_name),'matrix' +  str(run_num) +'.mat')
# data_=sio.loadmat(path_data_)
# matrix_data_=sio.loadmat(path__matrix_ )
# matrix_=matrix_data_['matrix']
# oa_=data_['oa']
# aa_=data_['aa']
# kappa_=data_['kappa']
#
# oa_[0][10]=oa[0][0]
# oa_[0][18]=oa[0][1]
# aa_[0][10]=aa[0][0]
# aa_[0][18]=aa[0][1]
# kappa_[0][10]=kappa[0][0]
# kappa_[0][18]=kappa[0][1]
# matrix_[10]=matrix[0]
# matrix_[18]=matrix[1]
# oa =oa_
# aa=aa_
# kappa=kappa_
# matrix =matrix_
# print(oa)
# sio.savemat(os.path.join(path, 'result' + str(run_num+1) + '.mat'),
#             {'oa': oa, 'aa': aa, 'kappa': kappa})
# sio.savemat(os.path.join(path, 'matrix' + str(run_num+1) + '.mat'),
#             {'matrix': matrix})
oa=np.reshape(oa,[oa.shape[1]])
aa=np.reshape(aa,[aa.shape[1]])
kappa=np.reshape(kappa,[kappa.shape[1]])
for i in range(len(matrix)):
    matrix_zero += matrix[i]
ac_list = []
for i in range(len(matrix_zero)):
    ac = matrix_zero[i, i] / sum(matrix_zero[:, i])
    ac_list.append(ac)
    #print('%0.2f'%(ac*100))
    print(i+1,'class:','%0.2f'%(ac*100))
ac=np.array(ac_list)
#print("aa: "+str(np.mean(ac)))
# m=0
# j=0
# k=0
# for i in range(oa.shape[0]):
#     if oa[i] <=0.99:
#         i+=1
#     if aa[i] <0.99:
#         j+=1
#         print("oa:"+str(oa[i])+"aa:"+str(aa[i])+"kappa:"+str(kappa[i]))
#     if kappa[i] <=0.99:
#         k+=1
#         print("------------")
#         print("oa:" + str(oa[i]) + "aa:" + str(aa[i]) + "kappa:" + str(kappa[i]))
#         print("------------")
# print("i:"+str(m)+"j:"+str(j)+"k:"+str(k))
print("the mean of oa:",'%0.4f'%(np.mean(oa)*100))
print("the mean of aa:",'%0.4f'%(np.mean(aa)*100))
print("the mean of kappa:",'%0.4f'%(np.mean(kappa)*100))
print("the mean of time:",'%0.4f'%(average_time),'min')