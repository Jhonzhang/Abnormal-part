import math
import scipy.stats as ss
from read_csv_common import *

def fig_actor_data(path_cvs, store_id_block_path):
    # 画图，只有actor
    type_ids, type_names, name_types, name_list = load_csv_info(path_cvs)  # 读取信息。
    target_cols_actors = type_ids['actor']  # 38个特征的id
    target_use_cols = target_cols_actors
    target_use_cols.insert(0, 0)  # 以时间为切分 #增加't_timstamp'
    df = pd.read_excel(path_cvs, skiprows=9, usecols=target_use_cols)  # 跳过前9行
    now_columns_name = list(df.columns)  # names
    some_time_blocks_keys, time_blocks_dict = load_useful_id_time_blocks(store_id_block_path)
    plt.switch_backend('agg')
    fig_store_path = 'fig3/' + 'target_cols_actors'
    if not os.path.exists(fig_store_path):
        os.makedirs(fig_store_path)
    print("新路径：",fig_store_path)
    this_block_ids,all_start_id_list = merge_each_time_block(some_time_blocks_keys, time_blocks_dict)
    print("序列长度：",len(this_block_ids))
    data = df.loc[this_block_ids] # 所有的数据合并 8*10（min）
    cnt_fig = 0  # 对每一列的数据画图
    fig_x_items = [i for i in range(len(this_block_ids))]
    for each_columns_name in now_columns_name[1:]:
        print("开始画图：",name_list.index(each_columns_name),each_columns_name,len(this_block_ids))
        each_column_data = data[each_columns_name].to_numpy()
        fig, ax = plt.subplots()
        #xticks = all_start_id_list #this_block_ids
        ymin = np.min(each_column_data)
        ymax = np.max(each_column_data)
        #ax.set_xticks(xticks)
        ax.xaxis.set_major_locator(MultipleLocator(661))
        ax.plot(fig_x_items, each_column_data)
        ax.vlines(all_start_id_list,ymin,ymax,linestyles = 'dashed',colors ='red')
        # print("观测：",each_column_data)
        # print('名称：',namestr_fun(target_use_cols))
        # each_column_data.plot()
        cnt_fig += 1
        fig_name = str(cnt_fig) + '_' + str(name_list.index(each_columns_name))+ '_' + each_columns_name + '.jpg'
        each_fig_path = fig_store_path + '/' + fig_name
        plt.savefig(each_fig_path)
        #plt.show()
        plt.close()

def cal_each_column_actor(each_column_data):
    #计算每列数据的周期性，即每种值出现的周期性统计。
    cycle_block_dict = {}
    tmp_list = []
    # hop_flag = True
    old_value = None
    cnt = 0
    this_column_data = each_column_data.tolist()
    N = len(this_column_data)
    for each_value in this_column_data:
        cnt += 1
        if each_value != old_value:
            if old_value == 'nan':
                print("特殊值：",old_value)
            if old_value not in cycle_block_dict:
                cycle_block_dict[old_value] = [len(tmp_list), ]  # 把旧值存储起来
            else:
                cycle_block_dict[old_value].append(len(tmp_list))
            old_value = each_value  # 重新赋值。
            tmp_list.clear() # 重置
            tmp_list.append(old_value)
            # hop_flag = False
            # pass
        else:
            tmp_list.append(each_value)
            if cnt == N:
                if old_value not in cycle_block_dict:
                    cycle_block_dict[old_value] = [len(tmp_list), ]  # 把旧值存储起来
                else:
                    cycle_block_dict[old_value].append(len(tmp_list))
    del cycle_block_dict[None]
    return cycle_block_dict

def cal_cycle_csv_data_actor(data, now_columns_name,name_list):
    # 计算actor周期
    all_data_dict = {}
    # 去除掉t_timestamp
    for each_columns_name in now_columns_name[1:]:
        each_column_data = data[each_columns_name].to_numpy()#数据：每个sensor的全部数据
        each_column_block_dict = cal_each_column_actor(each_column_data) # 计算周期性的运算
        if name_list.index(each_columns_name) ==5:
            each_column_block_dict = dict_slice(each_column_block_dict,-1)#第5列，不需要最后一位数据。

        all_data_dict = merge_dict(each_column_block_dict,all_data_dict)# 把更新后的each_column_block_dict返回
    # 人工校正周期
    final_cycle_dict = {0.0: []}
    for k, v in all_data_dict.items():
            each_v = list(v)
            each_v = sorted(each_v)
            if k in [2.0,1.0]:
                final_cycle_dict[k] = each_v #人工校正周期
            else:
                final_cycle_dict[0.0].extend(v) #人工校正周期
    special_value_list = sorted(list(final_cycle_dict[0.0])) # 只更新最后一步，因为该周期包含了很多非[2.0,1.0]周期的值
    final_cycle_dict[0.0] = special_value_list#人工校正周期
    # 计算上一步人工校正后每个周期的max,min,mean等值。
    actor_normal_final_cycle_dict = {}
    for k,v in final_cycle_dict.items():
        each_v_array = np.array(v[1:-1])
        max_value = np.max(each_v_array)
        min_value = np.min(each_v_array)
        mean_value = math.ceil(np.mean(each_v_array))
        #variance_value = math.ceil(np.var(each_v_array))
        median_value = math.ceil(np.median(each_v_array))#中位数
        mode_value = ss.mode(each_v_array)[0][0]#
        if k == 0.0:
            new_k = 'D3'#下降恢复
            actor_normal_final_cycle_dict[new_k] = {}
            max_v = mean_value
            min_v = mode_value
            actor_normal_final_cycle_dict[new_k]['max_v'] = max_v
            actor_normal_final_cycle_dict[new_k]['min_v'] = min_v
            actor_normal_final_cycle_dict[new_k]['mean'] = math.ceil((min_v + max_v)/2)
        elif k == 1.0:
            new_k = 'D1' #即正常时区
            actor_normal_final_cycle_dict[new_k] = {}
            max_v = max_value
            min_v = mode_value
            actor_normal_final_cycle_dict[new_k]['max_v'] = max_v
            actor_normal_final_cycle_dict[new_k]['min_v'] = min_v
            actor_normal_final_cycle_dict[new_k]['mean'] = math.ceil((min_v + max_v) / 2)
        elif k == 2.0:
            new_k = 'D2'#即上升时区时区
            actor_normal_final_cycle_dict[new_k] = {}
            max_v = mean_value
            min_v = mode_value
            actor_normal_final_cycle_dict[new_k]['max_v'] = max_v
            actor_normal_final_cycle_dict[new_k]['min_v'] = min_v
            actor_normal_final_cycle_dict[new_k]['mean'] = math.ceil((min_v + max_v) / 2)
    return actor_normal_final_cycle_dict

def cal_cycle_csv_data_actor_improved_mul(data, now_columns_name,name_list):
    # 计算actor周期
    data = pd.concat(data,axis=0)
    # print("合并后：",data.shape)
    all_data_dict = {}
    # 去除掉t_timestamp
    for each_columns_name in now_columns_name[1:]:
        each_column_data = data[each_columns_name].to_numpy()#数据：每个sensor的全部数据
        # print(each_columns_name,"列长度",len(each_column_data))
        each_column_block_dict = cal_each_column_actor(each_column_data) # 计算周期性的运算
        if name_list.index(each_columns_name) ==5:
            each_column_block_dict = dict_slice(each_column_block_dict,-1)#第5列，不需要最后一位数据。

        all_data_dict = merge_dict(each_column_block_dict,all_data_dict)# 把更新后的each_column_block_dict返回
    # 人工校正周期
    final_cycle_dict = {0.0: []}
    for k, v in all_data_dict.items():
            each_v = list(v)
            each_v = sorted(each_v)
            if k in [2.0,1.0]:
                final_cycle_dict[k] = each_v #人工校正周期
            else:
                final_cycle_dict[0.0].extend(v) #人工校正周期
    special_value_list = sorted(list(final_cycle_dict[0.0])) # 只更新最后一步，因为该周期包含了很多非[2.0,1.0]周期的值
    final_cycle_dict[0.0] = special_value_list#人工校正周期
    # 计算上一步人工校正后每个周期的max,min,mean等值。
    actor_normal_final_cycle_dict = {}
    for k,v in final_cycle_dict.items():
        each_v_array = np.array(v[1:-1])
        max_value = np.max(each_v_array)
        min_value = np.min(each_v_array)
        mean_value = math.ceil(np.mean(each_v_array))
        #variance_value = math.ceil(np.var(each_v_array))
        median_value = math.ceil(np.median(each_v_array))#中位数
        mode_value = ss.mode(each_v_array)[0][0]#
        if k == 0.0:
            new_k = 'D3'#下降恢复
            actor_normal_final_cycle_dict[new_k] = {}
            max_v = mean_value
            min_v = mode_value
            actor_normal_final_cycle_dict[new_k]['max_v'] = max_v
            actor_normal_final_cycle_dict[new_k]['min_v'] = min_v
            actor_normal_final_cycle_dict[new_k]['mean'] = math.ceil((min_v + max_v)/2)
        elif k == 1.0:
            new_k = 'D1' #即正常时区
            actor_normal_final_cycle_dict[new_k] = {}
            max_v = max_value
            min_v = mode_value
            actor_normal_final_cycle_dict[new_k]['max_v'] = max_v
            actor_normal_final_cycle_dict[new_k]['min_v'] = min_v
            actor_normal_final_cycle_dict[new_k]['mean'] = math.ceil((min_v + max_v) / 2)
        elif k == 2.0:
            new_k = 'D2'#即上升时区时区
            actor_normal_final_cycle_dict[new_k] = {}
            max_v = mean_value
            min_v = mode_value
            actor_normal_final_cycle_dict[new_k]['max_v'] = max_v
            actor_normal_final_cycle_dict[new_k]['min_v'] = min_v
            actor_normal_final_cycle_dict[new_k]['mean'] = math.ceil((min_v + max_v) / 2)
    return actor_normal_final_cycle_dict

def load_csv_data_actor_improve(path_cvs, store_id_block_path):
    """
    Args:
        path_cvs:包含actor和sensor数据的原始csv文件
        store_id_block_path:由cal_used_time_index()计算后存储每个时间区间对应id的路径。即存储有效id。

    Returns:周期，data_actor

    """
    type_ids, type_names, name_types, name_list = load_csv_info(path_cvs)  # 读取信息。
    # target_cols_actors = type_ids['actor']  # 81个特征的id
    target_cols_actors = [5, 12, 13, 15, 25, 33, 35, 36, 73]  # 根据每个传感前的画图结果，人工查看该列数据是否具有明显的周期性
    target_use_cols = target_cols_actors  # 使用哪些列
    target_use_cols.insert(0, 0)  # 以时间为切分 #增加't_timestamp'
    df = pd.read_excel(path_cvs, skiprows=9, usecols=target_use_cols)  # 跳过前9行，使用目标数据的id,带时间
    now_columns_name = list(df.columns)  # names
    some_time_blocks_keys, time_blocks_dict = load_useful_id_time_blocks(store_id_block_path)
    # plt.switch_backend('agg')
    this_block_ids, all_start_id_list = merge_each_time_block_improved(some_time_blocks_keys,time_blocks_dict)  # 合并所有数据块为一个总的数据集
    data = df.loc[this_block_ids]  # 所有的数据合并 8*10（min）
    actor_normal_final_cycle_dict = cal_cycle_csv_data_actor(data, now_columns_name,name_list)
    return actor_normal_final_cycle_dict,data

def load_csv_data_actor_improve_mul(path_cvs, store_id_block_path,all_block_len_list):
    """

    Args:
        path_cvs:包含actor和sensor数据的原始csv文件
        store_id_block_path:由cal_used_time_index()计算后存储每个时间区间对应id的路径。即存储有效id。

    Returns:周期，data_actor
    """
    type_ids, type_names, name_types, name_list = load_csv_info(path_cvs)  # 读取信息。
    # target_cols_actors = type_ids['actor']  # 81个特征的id
    target_cols_actors = [5, 12, 13, 15, 25, 33, 35, 36, 73]  # 根据每个传感前的画图结果，人工查看该列数据是否具有明显的周期性
    target_use_cols = target_cols_actors  # 使用哪些列
    target_use_cols.insert(0, 0)  # 以时间为切分 #增加't_timestamp'
    df = pd.read_excel(path_cvs, skiprows=9, usecols=target_use_cols)  # 跳过前9行，使用目标数据的id,带时间
    now_columns_name = list(df.columns)  # names
    some_time_blocks_keys, time_blocks_dict = load_useful_id_time_blocks(store_id_block_path)
    # plt.switch_backend('agg')
    # this_block_ids, all_start_id_list = merge_each_time_block_improved(some_time_blocks_keys,
    #                                                                   time_blocks_dict)  # 合并所有数据块为一个总的数据集
    data_pd_list = []
    for each_time_blocks_start_str in some_time_blocks_keys:
        this_index = some_time_blocks_keys.index(each_time_blocks_start_str)
        this_block_revise_len = all_block_len_list[this_index]
        this_block_ids = time_blocks_dict[each_time_blocks_start_str]  # 该区间对应的有效id
        this_block_data = df.loc[this_block_ids[0:this_block_revise_len]]  # 所有的数据合并 8*10（min）
        this_block_data = this_block_data.iloc[:, 1:]  # 去除timestamp
        # print("每个区间的开始时间：", each_time_blocks_start_str,len(this_block_ids[0:this_block_revise_len]))#校正后
        data_pd_list.append(this_block_data)
    # data = pd.DataFrame(data_list)
    data = data_pd_list
    actor_normal_final_cycle_dict = cal_cycle_csv_data_actor_improved_mul(data_pd_list, now_columns_name,name_list)
    return actor_normal_final_cycle_dict,data

def load_csv_data_actor_improve_mul_abnormal(path_cvs, store_id_block_path,all_block_len_list):
    """

    Args:
        path_cvs:包含actor和sensor数据的原始csv文件
        store_id_block_path:由cal_used_time_index()计算后存储每个时间区间对应id的路径。即存储有效id。

    Returns:周期，data_actor
    """
    type_ids, type_names, name_types, name_list = load_csv_info(path_cvs)  # 读取信息。
    # target_cols_actors = type_ids['actor']  # 81个特征的id
    target_cols_actors = [5, 12, 13, 15, 25, 33, 35, 36, 73]  # 根据每个传感前的画图结果，人工查看该列数据是否具有明显的周期性
    target_use_cols = target_cols_actors  # 使用哪些列
    target_use_cols.insert(0, 0)  # 以时间为切分 #增加't_timestamp'
    df = pd.read_excel(path_cvs, skiprows=9, usecols=target_use_cols)  # 跳过前9行，使用目标数据的id,带时间
    now_columns_name = list(df.columns)  # names
    some_time_blocks_keys, time_blocks_dict = load_useful_id_time_blocks(store_id_block_path)
    # plt.switch_backend('agg')
    # this_block_ids, all_start_id_list = merge_each_time_block_improved(some_time_blocks_keys,
    #                                                                   time_blocks_dict)  # 合并所有数据块为一个总的数据集
    data_pd_list = []
    for each_time_blocks_start_str in some_time_blocks_keys[1:]:
        this_index = some_time_blocks_keys.index(each_time_blocks_start_str)
        print("csv时的长度：",len(time_blocks_dict[each_time_blocks_start_str]))
        this_block_revise_len = all_block_len_list[this_index-1]
        this_block_ids = time_blocks_dict[each_time_blocks_start_str]  # 该区间对应的有效id
        this_block_data = df.loc[this_block_ids[0:this_block_revise_len]]  # 所有的数据合并 8*10（min）
        this_block_data = this_block_data.iloc[:, 1:]  # 去除timestamp
        # print("每个区间的开始时间：", each_time_blocks_start_str,len(this_block_ids[0:this_block_revise_len]))#校正后
        data_pd_list.append(this_block_data)
    # data = pd.DataFrame(data_list)
    data = data_pd_list
    actor_normal_final_cycle_dict = cal_cycle_csv_data_actor_improved_mul(data_pd_list, now_columns_name,name_list)
    return actor_normal_final_cycle_dict,data

if __name__ == "__main__":
    pcap = "/home/ztf/Downloads/A6/pcap/Dec2019_00013_20191206131500.pcap"
    path_traffic = "/home/ztf/Downloads/A6/pcap"
    path_cvs = "/home/ztf/Downloads/A6/csv/Dec2019.xlsx"
    store_id_block_path = "useful_id_blocks" # 存储计算好时间区间的id块。
    abnoraml_id_block_path = "abnormal_id_blocks"


    #测试read_actor数据读取。
    # all_block_len_list = [900, 601, 601, 601, 601, 660, 660, 660, 660, 301, 840]
    # actor_normal_final_cycle_dict,data = load_csv_data_actor_improve_mul(path_cvs, store_id_block_path,all_block_len_list)
    # print("actor周期：", actor_normal_final_cycle_dict, len(data))

    ## actor画图
    # fig_actor_data(path_cvs, store_id_block_path)
    all_block_len_list_abonormal = [300, 300, 300, 300, 180, 180, 180, 180, 180]
    actor_abnormal_final_cycle_dict,data = load_csv_data_actor_improve_mul_abnormal(path_cvs, abnoraml_id_block_path,all_block_len_list_abonormal)
    print("actor周期：", actor_abnormal_final_cycle_dict, len(data))