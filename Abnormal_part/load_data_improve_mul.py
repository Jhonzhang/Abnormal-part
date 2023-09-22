from read_actor import load_csv_data_actor_improve_mul,load_csv_data_actor_improve_mul_abnormal
from read_sensor import load_csv_data_sensor_improved_mul,load_csv_data_sensor_improved_mul_abnormal
from load_traffic_integrate_improved import load_traffic_scattered_feature_improved_mul,load_traffic_scattered_feature_improved_mul_abnormal
from load_traffic_integrate import load_traffic_scattered_feature
import pandas as pd
import numpy as np
from common_fun import read_pickle,store_data
from sklearn.impute import SimpleImputer

def load_data_mul(load_data_tmp_name,over_load = False):
    path_cvs = "/home/ztf/Downloads/A6/csv/Dec2019.xlsx"
    # load_csv_info(path_cvs)
    store_id_block_path = "useful_id_blocks"  # 存储计算好时间区间的id块。
    scattered_feature_path = '../../all_feature/network_traffic_scattered_feature'
    # step 1: load actor_data and cycle
    if over_load:

        # step 3： load traffic feature
        data_traffic, all_block_len_list = load_traffic_scattered_feature_improved_mul()

        durations, data_actor_pd_list = load_csv_data_actor_improve_mul(path_cvs, store_id_block_path,all_block_len_list)

        # step 2: load sensor data
        # data_sensor_pd = load_csv_data_sensor(path_cvs, store_id_block_path)
        data_sensor_pd_list,data_pd_index = load_csv_data_sensor_improved_mul(path_cvs, store_id_block_path, all_block_len_list)
        #print("sensor data :", len(data_sensor_pd))

        # data_traffic = load_traffic_scattered_feature(scattered_feature_path)
        #print("traffic data :", len(data_traffic))
        all_data = []
        for each_block_index in range(len(data_traffic)):
            each_block_data_traffic = data_traffic[each_block_index]
            each_pd_index = data_pd_index[each_block_index]
            each_block_data_traffic_pd = pd.DataFrame(each_block_data_traffic, index= each_pd_index)
            each_block_data_sensor_pd = data_sensor_pd_list[each_block_index]
            each_block_data_actor_pd = data_actor_pd_list[each_block_index]
            each_merge_data_pd = pd.concat([each_block_data_actor_pd, each_block_data_sensor_pd, each_block_data_traffic_pd], axis=1, join='outer')
            each_merge_data_pd = each_merge_data_pd.fillna(method='bfill')
            all_data.append(each_merge_data_pd.values.tolist())
            # pd_traffic_tmp_list.append(each_block_data_traffic_pd)
        # data_traffic_pd =  pd.DataFrame(pd_traffic_tmp_list)
        # data_actor_pd = data_actor_pd.iloc[:, 1:]  # 去除timestamp
        print("actor周期：", durations, len(data_sensor_pd_list))
        # data_sensor_pd  = data_sensor_pd.iloc[:,1:]#去除timestamp
        # data_sensor_pd = data_sensor_pd.round(3)
        # step concat data
        N = len(all_data)
        new_all_data = []
        for sub_all_data in all_data:
            new_all_data.extend(sub_all_data)
        all_data = new_all_data
        #all_data_pd2 = all_data_pd.iloc[:,1:].values.tolist()
        # all_data_pd2 = all_data_pd.values.tolist()
        all_data_np = np.array(all_data)  # list to array
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        
        all_data_np = imp.fit_transform(all_data_np)
        Or = all_data_np.tolist()
        # all_data_np = [imp.fit_transform(sub_all_data) for sub_all_data in all_data]
        # Or = [sub_all_data.tolist() for sub_all_data in all_data_np]
        # print(len(all_data_pd2[0]),all_data_pd2[0])
        store_data([Or,durations,N],load_data_tmp_name)
        print("训练集规模：", type(Or), len(Or),len(Or[0]),N)
        return Or,durations,N
    # else:
    #     all_data, durations = read_pickle(load_data_tmp_name)
    #     N = len(all_data)
    #     new_all_data = []
    #     for sub_all_data in all_data:
    #         new_all_data.extend(sub_all_data)
    #     # print(all_data)
    #     # print("数据规模：", type(all_data), len(all_data[0][0]))
    #     all_data_np = np.array(new_all_data)  # list to array
    #     imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    #     all_data_np = imp.fit_transform(all_data_np)
    #     Or = all_data_np.tolist()
    #     print("数据规模2：", type(Or), len(Or),len(new_all_data),all_data_np.shape)
    #     return Or, durations,N
    else:
        Or, durations,N = read_pickle(load_data_tmp_name)
        print("训练集规模：", type(Or), len(Or),len(Or[0]))
        return Or, durations,N

def load_data_mul_abnormal(load_data_tmp_name,over_load = False):
    path_cvs = "~/project_ztf/csv/Dec2019.xlsx"
    store_id_block_path = "abnormal_id_blocks"  # 存储计算好时间区间的id块。
    store_all_feature_path_abnormal = '/home/ztf/Abnormal_part/all_feature_abnormal' #存储切分好的特征存放路径,每个meta_pcap的数据包长度序列。

    if over_load:
        # step 1： load traffic feature,data_traffic是原始的数据包长度序列，没有进行统计提取
        data_traffic, all_block_len_list = load_traffic_scattered_feature_improved_mul_abnormal(store_all_feature_path_abnormal)
        # step 2: load actor_data and cycle
        durations, data_actor_pd_list = load_csv_data_actor_improve_mul_abnormal(path_cvs, store_id_block_path,all_block_len_list)
        # step 3: load sensor data
        data_sensor_pd_list,data_pd_index = load_csv_data_sensor_improved_mul_abnormal(path_cvs, store_id_block_path, all_block_len_list)
        all_data = []
        for each_block_index in range(len(data_traffic)):
            each_block_data_traffic = data_traffic[each_block_index]
            each_pd_index = data_pd_index[each_block_index]
            each_block_data_traffic_pd = pd.DataFrame(each_block_data_traffic, index= each_pd_index)
            each_block_data_sensor_pd = data_sensor_pd_list[each_block_index]
            each_block_data_actor_pd = data_actor_pd_list[each_block_index]
            each_merge_data_pd = pd.concat([each_block_data_actor_pd, each_block_data_sensor_pd, each_block_data_traffic_pd], axis=1, join='outer')
            each_merge_data_pd = each_merge_data_pd.fillna(method='bfill')
            all_data.append(each_merge_data_pd.values.tolist())

        print("actor周期：", durations, len(data_sensor_pd_list))
        new_all_data = []
        N = len(all_data)
        for sub_all_data in all_data:
            new_all_data.extend(sub_all_data)
        all_data_np = np.array(new_all_data)  # list to array
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')

        # all_data_np = [imp.fit_transform(sub_all_data) for sub_all_data in all_data]
        # Or = [sub_all_data.tolist() for sub_all_data in all_data_np]
        all_data_np = imp.fit_transform(all_data_np)
        Or = all_data_np.tolist()
        # print(len(all_data_pd2[0]),all_data_pd2[0])
        store_data([Or, durations, N], load_data_tmp_name)
        return Or, durations, N
        # all_data = np.array(all_data)  # list to array
        # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        #
        # all_data_np = [imp.fit_transform(sub_all_data) for sub_all_data in all_data]
        # Or = [sub_all_data.tolist() for sub_all_data in all_data_np]
        # N = len(all_data)
        # store_data([Or,durations,N],load_data_tmp_name)
        # return Or,durations,N
    else:
        Or, durations ,N = read_pickle(load_data_tmp_name)
        print("异常数据规模：", type(Or), len(Or),len(Or[0]))
        return Or, durations,N

if __name__ == "__main__":
    # 1 normal part
    # load_data_tmp_name = 'load_data_tmp_improved_mul'
    # all_data_pd2, durations = load_data_mul(load_data_tmp_name,over_load=False)
    # print(durations)
    
    # 2 adnormal part
    load_data_tmp_name_abnormal = 'load_data_tmp_improved_mul_abnormal_some_column'
    all_data_pd_abnormal, duration_abnormal,N = load_data_mul_abnormal(load_data_tmp_name_abnormal,over_load=False)
    print(len(all_data_pd_abnormal),duration_abnormal,N,len(all_data_pd_abnormal[0]))





