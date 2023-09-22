from read_csv_common import *

def fig_sensor_data(path_cvs, store_id_block_path):
    #画图 sensor的周期数据图。
    type_ids, type_names, name_types, name_list = load_csv_info(path_cvs)  # 读取信息。
    target_cols_sensors = type_ids['sensor']# 28个特征的id
    target_use_cols = target_cols_sensors
    target_use_cols.insert(0, 0)  # 以时间为切分 #增加't_timestamp'
    df = pd.read_excel(path_cvs, skiprows=9, usecols=target_use_cols)  # 跳过前9行
    now_columns_name = list(df.columns)  # names
    some_time_blocks_keys, time_blocks_dict = load_useful_id_time_blocks(store_id_block_path)
    plt.switch_backend('agg')
    fig_store_path = 'fig3/' + 'target_cols_sensors'
    if not os.path.exists(fig_store_path):
        os.makedirs(fig_store_path)
    print("新路径：",fig_store_path)
    # for each_time_blocks in some_time_blocks_keys:
    #     this_block_ids = time_blocks_dict[each_time_blocks]# 该区间对应的有效id
        # data = df.loc[this_block_ids].values
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

def cal_each_column_sensor(each_column_data):
    # 刻画连续数据的周期性
    cycle_block_dict = {}
    tmp_list = []
    #hop_flag = True
    old_value = None
    cnt = 0
    all_data = each_column_data.tolist()
    N = len(all_data)
    for each_value in each_column_data.tolist():
        cnt += 1
        if each_value != old_value:
            if old_value == 'nan':
                print("特殊值：",old_value)
            if old_value not in cycle_block_dict:
                cycle_block_dict[old_value] = [len(tmp_list), ]  # 把旧值存储起来
            else:
                cycle_block_dict[old_value].append(len(tmp_list))
            old_value = each_value  # 重新赋值。
            tmp_list.clear()  # 清空，重置
            tmp_list.append(old_value)
            #hop_flag = False
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

def view_each_column_sensor(fig_store_path,each_columns_name,cnt_fig,each_column_data,all_start_id_list,fig_x_items,this_index):
    # 刻画连续数据的周期性
    fig, ax = plt.subplots()
    # xticks = all_start_id_list #this_block_ids
    ymin = np.min(each_column_data)
    ymax = np.max(each_column_data)
    # ax.set_xticks(xticks)
    ax.xaxis.set_major_locator(MultipleLocator(661))
    ax.plot(fig_x_items, each_column_data)
    ax.vlines(all_start_id_list, ymin, ymax, linestyles='dashed', colors='red')
    # print("观测：",each_column_data)
    # print('名称：',namestr_fun(target_use_cols))
    # each_column_data.plot()
    cnt_fig += 1
    fig_name = str(cnt_fig) + '_' +  str(this_index) +'_' + each_columns_name + '.jpg'
    each_fig_path = fig_store_path + '/' + fig_name
    plt.savefig(each_fig_path)
    # plt.show()
    plt.close()

def cal_cycle_csv_data_sensor(data):
    # 计算sensor周期

    all_data_dict = {}
    # for each_columns_name in now_columns_name[1:]:
    #     # print("当前计算在值actor：", name_list.index(each_columns_name),each_columns_name, len(this_block_ids))
    #     # this_index = name_list.index(each_columns_name)
    #     # each_column_data = data[each_columns_name].to_numpy()  # 数据：每个sensor的全部数据
    #     # df = data[each_columns_name]
    #     # tmp_dict = {}
    #     #each_column_block_dict = cal_each_column_actor(each_column_data)
    #     # print("each_column_block_dict",each_column_block_dict)
    #     # if name_list.index(each_columns_name) == 5:
    #     #     each_column_block_dict = dict_slice(each_column_block_dict, -1)
    #     print("特殊值：",each_columns_name)
    #     #view_each_column_sensor(fig_store_path,each_columns_name,cnt_fig,each_column_data,all_start_id_list,fig_x_items,this_index)
    #     # ts = each_colum_data_dataframe
    #     diff_smooth_fig(fig_store_path_diff, df, each_columns_name, cnt_fig, all_start_id_list, fig_x_items)
    #     cnt_fig += 1
    #     #all_data_dict = merge_dict(each_column_block_dict, all_data_dict)  # 把更新后的each_column_block_dict返回
    #     #print("特殊观测：",Counter(each_column_data.tolist()))
    print("全部的值：",len(data))

    # 人工校正
    #  周期
    # print("最终的dict:",all_data_dict)
    # final_cycle_dict = {0.0: []}
    # # for k,v in all_data_dict.items():
    # #     each_v = list(v)
    # #     each_v = sorted(each_v)
    # #     print("k:",k," v:",each_v)
    # for k, v in all_data_dict.items():
    #     each_v = list(v)
    #     each_v = sorted(each_v)
    #     if k in [2.0, 1.0]:
    #         final_cycle_dict[k] = each_v
    #     else:
    #         final_cycle_dict[0.0].extend(v)
    # special_value_list = sorted(list(final_cycle_dict[0.0]))  # 只更新最后一步
    # final_cycle_dict[0.0] = special_value_list
    # # final_cycle_dict = sorted(final_cycle_dict.items(),key=lambda d:d[1], reverse = False)
    # print("最后的结果:", final_cycle_dict)
    # actor_normal_final_cycle_dict = {}
    # for k, v in final_cycle_dict.items():
    #     each_v_array = np.array(v[1:-1])
    #
    #     max_value = np.max(each_v_array)
    #     min_value = np.min(each_v_array)
    #     mean_value = math.ceil(np.mean(each_v_array))
    #     # variance_value = math.ceil(np.var(each_v_array))
    #     median_value = math.ceil(np.median(each_v_array))  # 中位数
    #     mode_value = ss.mode(each_v_array)[0][0]  #
    #     if k == 0.0:
    #         new_k = 'd3'  # 下降恢复
    #         actor_normal_final_cycle_dict[new_k] = {}
    #         max_v = mean_value
    #         min_v = mode_value
    #         actor_normal_final_cycle_dict[new_k]['max_v'] = max_v
    #         actor_normal_final_cycle_dict[new_k]['min_v'] = min_v
    #     elif k == 1.0:
    #         new_k = 'd1'  # 即正常时区
    #         actor_normal_final_cycle_dict[new_k] = {}
    #         max_v = max_value
    #         min_v = mode_value
    #         actor_normal_final_cycle_dict[new_k]['max_v'] = max_v
    #         actor_normal_final_cycle_dict[new_k]['min_v'] = min_v
    #     elif k == 2.0:
    #         new_k = 'd2'  # 即上升时区时区
    #         actor_normal_final_cycle_dict[new_k] = {}
    #         max_v = mean_value
    #         min_v = mode_value
    #         actor_normal_final_cycle_dict[new_k]['max_v'] = max_v
    #         actor_normal_final_cycle_dict[new_k]['min_v'] = min_v
    #     print(k, "观测的值actor：", max_value, min_value, mean_value, median_value, mode_value)
    # print("规则化后的值：",actor_normal_final_cycle_dict)
    #return actor_normal_final_cycle_dict

def load_csv_data_sensor(path_cvs, store_id_block_path):
    """
    Args:
        path_cvs: 包含actor和sensor数据的原始csv文件
        store_id_block_path:由cal_used_time_index()计算后存储每个时间区间对应id的路径。即存储有效id。

    Returns:DataFram格式的data_sensor

    """
    target_cols_sensors_0 = [3,8,30,31]  # 根据每个传感器的画图结果，人工查看该列数据是否具有明显的周期性，这几列的数据与actor一样，规整的几字型图
    target_cols_sensors_1 = [2,9,10,11,57,58,62]  # ?清晰，不密集
    target_cols_sensors_2 = [59,60,61]#清晰，一般密集，？_2,3
    target_cols_sensors_best = [26, 27, 29] # 明显周期性
    target_cols_sensors_4 = [42,44,52,53,54,56]#密集
    target_cols_sensors_all = target_cols_sensors_0 + target_cols_sensors_1 + target_cols_sensors_2 + target_cols_sensors_best + target_cols_sensors_4
    target_use_cols_org = target_cols_sensors_all  # 使用哪些列
    target_use_cols = target_use_cols_org.copy()
    target_use_cols.insert(0, 0)  # 以时间为切分 #增加't_timestamp'
    df = pd.read_excel(path_cvs, skiprows=9, usecols=target_use_cols)  # 跳过前9行，使用目标数据的id
    # now_columns_name = list(df.columns)  # names
    some_time_blocks_keys, time_blocks_dict = load_useful_id_time_blocks(store_id_block_path)
    #plt.switch_backend('agg')
    fig_store_path = 'fig_sensor/target_cols_sensors_some/' + 'target_cols_sensors_all'
    if not os.path.exists(fig_store_path):
        os.makedirs(fig_store_path)
    this_block_ids, all_start_id_list = merge_each_time_block_improved(some_time_blocks_keys,time_blocks_dict)  # 合并所有数据块为一个总的数据集
    data = df.loc[this_block_ids]  # 所有的数据合并 8*10（min）,在load_data.py中去除t_timestamp
    return data

def load_csv_data_sensor_improved_mul(path_cvs, store_id_block_path,all_block_len_list):
    """
    Args:
        path_cvs: 包含actor和sensor数据的原始csv文件
        store_id_block_path:由cal_used_time_index()计算后存储每个时间区间对应id的路径。即存储有效id。

    Returns:DataFram格式的data_sensor

    """
    type_ids, type_names, name_types, name_list = load_csv_info(path_cvs)  # 读取信息。
    target_cols_sensors_0 = [3,8,30,31]  # 根据每个传感器的画图结果，人工查看该列数据是否具有明显的周期性，这几列的数据与actor一样，规整的几字型图
    target_cols_sensors_1 = [2,9,10,11,57,58,62]  # ?清晰，不密集
    target_cols_sensors_2 = [59,60,61]#清晰，一般密集，？_2,3
    target_cols_sensors_best = [26, 27, 29] # 明显周期性
    target_cols_sensors_4 = [42,44,52,53,54,56]#密集
    target_cols_sensors_all = target_cols_sensors_0 + target_cols_sensors_1 + target_cols_sensors_2 + target_cols_sensors_best + target_cols_sensors_4
    target_use_cols_org = target_cols_sensors_all  # 使用哪些列
    target_use_cols = target_use_cols_org.copy()
    target_use_cols.insert(0, 0)  # 以时间为切分 #增加't_timestamp'
    df = pd.read_excel(path_cvs, skiprows=9, usecols=target_use_cols)  # 跳过前9行，使用目标数据的id
    # now_columns_name = list(df.columns)  # names
    some_time_blocks_keys, time_blocks_dict = load_useful_id_time_blocks(store_id_block_path)
    #plt.switch_backend('agg')
    fig_store_path = 'fig_sensor/target_cols_sensors_some/' + 'target_cols_sensors_all'
    if not os.path.exists(fig_store_path):
        os.makedirs(fig_store_path)

    data_pd_list = []
    data_pd_index = []
    for each_time_blocks_start_str in some_time_blocks_keys:
        this_index = some_time_blocks_keys.index(each_time_blocks_start_str)
        this_block_revise_len = all_block_len_list[this_index]
        this_block_ids = time_blocks_dict[each_time_blocks_start_str]  # 该区间对应的有效id
        this_block_data = df.loc[this_block_ids[0:this_block_revise_len]]  # 所有的数据合并 8*10（min）
        # print("每个区间的开始时间：", each_time_blocks_start_str, len(this_block_ids[0:this_block_revise_len]))  # 校正后
        this_block_data = this_block_data.iloc[:, 1:]  # 去除timestamp
        data_pd_list.append(this_block_data.round(3))
        data_pd_index.append(this_block_data.index)
    # data = pd.DataFrame(data_list)
    data = data_pd_list
    # this_block_ids, all_start_id_list = merge_each_time_block_improved(some_time_blocks_keys,time_blocks_dict)  # 合并所有数据块为一个总的数据集
    # data = df.loc[this_block_ids]  # 所有的数据合并 8*10（min）,在load_data.py中去除t_timestamp
    return data,data_pd_index

def load_csv_data_sensor_improved_mul_abnormal(path_cvs, store_id_block_path,all_block_len_list):
    """
    Args:
        path_cvs: 包含actor和sensor数据的原始csv文件
        store_id_block_path:由cal_used_time_index()计算后存储每个时间区间对应id的路径。即存储有效id。
    Returns:DataFram格式的data_sensor

    """
    type_ids, type_names, name_types, name_list = load_csv_info(path_cvs)  # 读取信息。
    #target_cols_sensors_all = type_ids['sensor']  # 81个特征的id
    target_cols_sensors_0 = [3,8,30,31]  # 根据每个传感器的画图结果，人工查看该列数据是否具有明显的周期性，这几列的数据与actor一样，规整的几字型图
    target_cols_sensors_1 = [2,9,10,11,57,58,62]  # ?清晰，不密集
    target_cols_sensors_2 = [59,60,61]#清晰，一般密集，？_2,3
    target_cols_sensors_best = [26, 27, 29] # 明显周期性
    target_cols_sensors_4 = [42,44,52,53,54,56]#密集
    target_cols_sensors_all = target_cols_sensors_0 + target_cols_sensors_1 + target_cols_sensors_2 + target_cols_sensors_best + target_cols_sensors_4
    target_use_cols_org = target_cols_sensors_all  # 使用哪些列
    target_use_cols = target_use_cols_org.copy()
    target_use_cols.insert(0, 0)  # 以时间为切分 #增加't_timestamp'
    df = pd.read_excel(path_cvs, skiprows=9, usecols=target_use_cols)  # 跳过前9行，使用目标数据的id
    # now_columns_name = list(df.columns)  # names
    some_time_blocks_keys, time_blocks_dict = load_useful_id_time_blocks(store_id_block_path)
    #plt.switch_backend('agg')
    # fig_store_path = 'fig_sensor/target_cols_sensors_some/' + 'target_cols_sensors_all'
    # if not os.path.exists(fig_store_path):
    #     os.makedirs(fig_store_path)
    data_pd_list = []
    data_pd_index = []
    for each_time_blocks_start_str in some_time_blocks_keys[1:]:
        this_index = some_time_blocks_keys.index(each_time_blocks_start_str)
        this_block_revise_len = all_block_len_list[this_index-1]
        this_block_ids = time_blocks_dict[each_time_blocks_start_str]  # 该区间对应的有效id
        this_block_data = df.loc[this_block_ids[0:this_block_revise_len]]  # 所有的数据合并 8*10（min）
        # print("每个区间的开始时间：", each_time_blocks_start_str, len(this_block_ids[0:this_block_revise_len]))  # 校正后
        this_block_data = this_block_data.iloc[:, 1:]  # 去除timestamp
        data_pd_list.append(this_block_data.round(3))
        data_pd_index.append(this_block_data.index)
    # data = pd.DataFrame(data_list)
    data = data_pd_list
    # this_block_ids, all_start_id_list = merge_each_time_block_improved(some_time_blocks_keys,time_blocks_dict)  # 合并所有数据块为一个总的数据集
    # data = df.loc[this_block_ids]  # 所有的数据合并 8*10（min）,在load_data.py中去除t_timestamp
    return data,data_pd_index

if __name__ == "__main__":
    pcap = "/home/ztf/Downloads/A6/pcap/Dec2019_00013_20191206131500.pcap"
    path_traffic = "/home/ztf/Downloads/A6/pcap"
    path_cvs = "/home/ztf/Downloads/A6/csv/Dec2019.xlsx"
    store_id_block_path = "useful_id_blocks" # 存储计算好时间区间的id块。
    ##
    # all_block_len_list =  [900, 601, 601, 601, 601, 660, 660, 660, 660, 301, 840]
    # data,data_pd_index = load_csv_data_sensor_improved_mul(path_cvs, store_id_block_path,all_block_len_list)
    # print("sensor：", len(data))
    # fig_sensor_data(path_cvs, store_id_block_path)
    
    abnoraml_id_block_path = "abnormal_id_blocks"
    all_block_len_list_abonormal = [300, 300, 300, 300, 180, 180, 180, 180, 180]
    actor_abnormal_final_cycle_dict_abnormal,data_abnormal = load_csv_data_sensor_improved_mul_abnormal(path_cvs, abnoraml_id_block_path,all_block_len_list_abonormal)
    print("actor周期：", actor_abnormal_final_cycle_dict_abnormal, len(data_abnormal))


