from common_time import now_time_str
import matlab.engine
from load_data_improve_mul import *
from load_data_improve2 import *
from load_data_without_traffic import *
import numpy as np
import math
from likelihood_distribution_view_one import fig_one_diagram
# sys.path.append('../')
np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf)
import random
# from sklearn.model_selection import train_test_split
def comprise_np_to_list(np_datasets):
    new_list_dataset = []
    for each_np_data in np_datasets:
        each_data = each_np_data.tolist()
        new_list_dataset.extend(each_data)
    return new_list_dataset
if __name__ == "__main__":
    """只把真实数据，输入到generate_test_seq中，不用python重写其功能函数。"""
    eng = matlab.engine.start_matlab()  # start matlab env
    eng.cd(r'/home/ztf/Abnormal_part', nargout=0)  # 这个文件夹很重要啊！一定要是当前路径
    for i in range(2):
        retraining_bool = 1 #True  # 是否重写训练HsMM
        if retraining_bool:
            # print("START training! :", now_time_str())
            # load normal data
            # load_data_tmp_name = 'load_data_without_traffic_dat'
            load_normal_dataset = 'normal_dataset_np'
            Or, durations, N = load_data_mul2(load_normal_dataset, over_load=False)
            # print("加载数据集完成！")
            # Or, durations, N = load_data_mul_without_traffic(load_data_tmp_name, over_load=False)
            # Or的预处理放在load函数中，直接输出matlab可用的数据格式。
            # matlab无法直接使用python数组，需要使用double方法转
            # N_test = math.floor(0.1*N) # 测试样本的个数
            # N_train = N - N_test # 训练集个数
            
            MT = [900, 601, 601, 601, 601, 660, 660, 660, 660, 301, 840]  # length of each sub-sequence
            MT_index = list(range(N)) #产生序列的索引
            # MT_index_copy = MT_index
            for MT_test_index in MT_index[0:]:
                i_th = MT_index.index(MT_test_index)
                print("---------This is the "+ str(i_th + 1) + " 'normal training!----------ztf5-")
                MT_index_copy = MT_index.copy() # 每次建立一个副本
                MT_index_copy.pop(MT_test_index) #剔除测试集的索引
                MT_train_index = MT_index_copy # 训练集索引
                N_test = 1
                N_train = N - N_test
                # y = MT_index
                # X = Or
                # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) 
                # MT_test_index = random.sample(MT_index,N_test)# 产生测试序列的索引集合
                # MT_train_index = [i for i in MT_index if i not in MT_test_index]# 训练集的数据 
                MT_train = [MT[index] for index in MT_train_index] # 训练集序列的长度集合
                # MT_test = [MT[index] for index in MT_test_index] # 测试集序列的长度集合
                MT_test = [MT[MT_test_index]] #因为只有一个值，只能以此索引
                Or_train = comprise_np_to_list([Or[index] for index in MT_train_index])#训练集
                # Or_test =  comprise_np_to_list([Or[index] for index in MT_test_index]) #测试集
                Or_test = Or[MT_test_index]
                MT_train = matlab.double(MT_train) 
                MT_test = matlab.double(MT_test)
                # N = 11
                Or_train = matlab.double(Or_train)  # transform numpy array to matlab doubl 
                Dim = len(Or_train[0])  # 47维
                # print(type(Or_train),"序列总长：",len(Or_train),"维度:",Dim,"子序列：",N_train)
                Or_test = matlab.double(Or_test)  # transform numpy array to matlab doubl 
                D = 5
                T0 = 900
                
                load_abnormal_dataset = 'abnormal_dataset_np'
                Or_abnormal, durations_abnormal, N_abnormal =  load_data_mul_abnormal2(load_abnormal_dataset, over_load=False)
                MT_abnormal = [300, 300, 300, 300, 180, 180, 180, 180, 180]
                MT_abnormal = matlab.double(MT_abnormal)
                Or_abnormal = comprise_np_to_list(Or_abnormal)
                # print(type(Or_abnormal))
                Or_abnormal = matlab.double(Or_abnormal)  # transform numpy array to matlab doubl 
                Dim_abnormal = len(Or_abnormal[0])  # 47维
                # print(type(Or_abnormal),"序列总长：",len(Or_abnormal),"维度:",Dim,"子序列：",N_abnormal)
                # Or_abnormal = matlab.double(Or_abnormal)
                T0_abnormal = 300
                D_abnormal = 5
                retrain_traing_dataset = 1
                vertify = 0
                if vertify:
                    #Dim, T0, N, Or, MT, D, Or_test, MT_test, Dim_test, T0_test, N_test, D_test
                    ret1 = eng.generate_train_seq_vertify(Dim,T0,N_train,Or_train, MT_train, D, Or_abnormal, MT_abnormal, Dim_abnormal, T0_abnormal, N_abnormal, D_abnormal,nargout=2)
                    [Loglikelihood_normal, Loglikelihood_abnormal] = ret1
                    fig_one_diagram(Loglikelihood_normal, Loglikelihood_abnormal,MT_test_index,combinnation_order = 2)
                else:
                    ret2 = eng.generate_train_seq4f_2(Dim,T0,N_train,Or_train, MT_train,D,Or_abnormal, MT_abnormal, Dim_abnormal, T0_abnormal, N_abnormal, D_abnormal,Or_test, MT_test,N_test,nargout=1)
                    best_fitness = ret2
                
                # break #测试，只运行一次。
        break #只运行一次，不用多次测试
    eng.quit()  # stop matalb evn
