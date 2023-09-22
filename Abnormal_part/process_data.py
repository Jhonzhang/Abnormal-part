from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
import os
# from torch.utils.data import DataLoader, TensorDataset
# import torch
import os

def sliding_window(df_sequence, window_size, stride):
        num_windows = (len(df_sequence) - window_size) // stride + 1
        # print("窗口数：",num_windows)
        windows = []

        for i in range(num_windows):
            start = i * stride
            end = start + window_size
            window = df_sequence[start:end] #读取特定的行
            windows.append(window)
        print("切分出序列个数：",len(windows))
        return windows

# 处理输出，删除某些列，并分成单独的传感器
def create_data(path, deal_path, window_size, stride,normalization = False):
    # deal_path = 'datasets/SWaT/'
    if 'swat' in path.lower():
        df_train = pd.read_csv(path + "/SWaT_Dataset_Normal_v3.csv")
        df_train = df_train.drop([' Timestamp', 'Normal/Attack'], axis=1)
        # df_train = df_train.drop(['AIT201','AIT202','AIT203','P201','AIT401','AIT402','AIT501','AIT502','AIT503','AIT504','FIT503','FIT504','PIT501','PIT502','PIT503'], axis=1)
        # df_train = df_train.drop(['AIT201','P201','FIT601','P601','P602','P603'],axis=1)

        df_test = pd.read_csv(path + "/SWaT_Dataset_Attack_v0.csv")
        labels = [float(label != 'Normal') for label in df_test.iloc[:, -1].values]
        df_test = df_test.drop([' Timestamp', 'Normal/Attack'], axis=1)
        # df_test = df_test.drop(
        #     ['AIT201', 'AIT202', 'AIT203', 'P201', 'AIT401', 'AIT402', 'AIT501', 'AIT502', 'AIT503', 'AIT504', 'FIT503',
        #      'FIT504', 'PIT501', 'PIT502', 'PIT503'], axis=1)
        # df_test = df_test.drop(['AIT201', 'P201', 'FIT601', 'P601', 'P602', 'P603'],axis=1)

        # os.chdir('/data/lanbin') #用于改变当前工作目录到指定的路径。
        # 执行器可以onehot编码，变成多维数据
        if normalization:
            train_output_path = deal_path + 'train_normal'
            test_output_path = deal_path + 'test_normal'
        else:
            train_output_path = deal_path + 'train'
            test_output_path = deal_path + 'test'
        if not os.path.exists(train_output_path):
            os.makedirs(train_output_path)
        if not os.path.exists(test_output_path):
            os.makedirs(test_output_path)
        print(type(df_train),df_train.shape)
        print(type(df_test),df_test.shape)
        train_data = sliding_window(df_train, window_size, stride)
        test_data = sliding_window(df_test, window_size, stride)
        # labels = np.load(os.path.join(test_path, f'labels.npy'))
        np.save(os.path.join(train_output_path, f'train_data_{str(window_size)}.npy'), train_data)
        np.save(os.path.join(test_output_path, f'test_data_{str(window_size)}.npy'), test_data)

    # for i in range(df_train.shape[0]):
    #     this_row_data = df_train.iloc[i]
    #     print(i,len(this_row_data.values),this_row_data,this_row_data.values)
    #     break
    
    # # for idx, name in enumerate(df_train):
    #     print(name)
    #     train_data = df_train[name].values.reshape(-1, 1) #列，-1被理解为unspecified value，意思是未指定为给定的
    #     # test_data = df_test[name].values.reshape(-1, 1)
    #     stdsc = MinMaxScaler()
    #     train_data = stdsc.fit_transform(train_data)
    #     test_data = stdsc.transform(test_data)
    #     np.save(os.path.join(train_output_path, f'{idx}.npy'), train_data)
    #     np.save(os.path.join(test_output_path, f'{idx}.npy'), test_data)

    # labels = np.array(labels).reshape(-1, 1)
    # np.save(os.path.join(test_output_path, f'labels.npy'), labels)
    


def load_data(name, deal_path,window_size):
    if os.path.exists(deal_path + 'train/train_data_100.npy'):
        train_path = deal_path + 'train'
        test_path = deal_path + 'test'
    else:
        train_path = deal_path + 'train_normal'
        test_path = deal_path + 'test_normal'
    train_loader = []
    test_loader = []
    train_loader = np.load(os.path.join(train_path, f'train_data_{str(window_size)}.npy'))
    test_loader = np.load(os.path.join(test_path, f'test_data_{str(window_size)}.npy'))
    # for file in range(len(os.listdir(train_path))):
    #     data = np.load(os.path.join(train_path, f'{file}.npy'))
    #     # loader = DataLoader(data, batch_size=data.shape[0])
    #     train_loader.append(data)
    # 有label文件所以-1
    # for file in range(len(os.listdir(test_path))-1):
    #     data = np.load(os.path.join(test_path, f'{file}.npy'))
    #     # loader = DataLoader(data, batch_size=data.shape[0])
    #     test_loader.append(data)
        # loader = [i[:, debug:debug+1] for i in loader]
    # if args.less: loader[0] = cut_array(0.2, loader[0])
    # print(type(loader[0]), type(loader[1]), type(loader[2]))
    # labels = np.load(os.path.join(test_path, f'labels.npy'))
    print("训练样本数："+str(len(train_loader)),"测试样本数："+str(len(test_loader)))
    return train_loader, test_loader



if __name__=='__main__':
    dataset_root_path = "/home/ztf/data/A1/" #数据集所在的根目录
    dataset_name = 'SWaT'
    # dataset_name = 'BATADAL'
    # dataset_name = 'WADI'
    path = dataset_root_path + dataset_name
    deal_path = dataset_root_path + 'datasets/' + dataset_name + '/'
    window_size = 100 #窗口大小
    suffix = '2w'
    stride = 100 #步长，无重叠时，等于窗口大小
    #step1 处理原始数据，只需运行一次
    # create_data(path, deal_path, window_size, stride, normalization=True)
    # print("Deal train test labels Done.")

    train_loader, test_loader = load_data(dataset_name, deal_path,window_size)
    print("Read train_loader test_loader labels Done.")

    # windows_path = os.path.join(deal_path, f'{windows_size}_size_{suffix}')
    # labels_path = os.path.join(deal_path, f'labels_{windows_size}_{suffix}')
    # if os.path.exists(windows_path):
    #     pass
    # else:
    #     os.mkdir(windows_path)
    # if os.path.exists(labels_path):
    #     pass
    # else:
    #     os.mkdir(labels_path)
    # train_windows = []
    # test_windows = []
    # # lan是按每个维度进行训练！
    # for i in range(len(train_loader)):
    #     trainD, testD = next(iter(train_loader[i])), next(iter(test_loader[i]))
    #     trainD, testD = convert_to_windows(trainD, windows_size), convert_to_windows(testD, windows_size)
    #     train_windows.append(trainD)
    #     np.save(os.path.join(windows_path, f'{i}_train.npy'), trainD)
    #     test_windows.append(testD)
    #     np.save(os.path.join(windows_path, f'{i}_test.npy'), testD)
    # labelsD = convert_to_windows_labels(labels, windows_size, labels_path)
    # print(len(train_windows), len(test_windows))