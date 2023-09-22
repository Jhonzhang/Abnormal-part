import statsmodels.api as sm
from scipy.fftpack import fft,fftfreq
import pandas as pd
from common_fun import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import io

def read_df_info(path_cvs, over_read: bool = False):
    df_info_path = "df_info.txt"
    if over_read:
        df = pd.read_excel(path_cvs, skiprows=9)  # 跳过前9行
        # with open(csv_file, 'r', newline='', encoding='utf-8') as f:
        # df.head() # 只输出前行。
        #     reader = csv.reader(f)
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()  # 存储为字符串
        # print("获取到的信息:",type(s),s)
        with open(df_info_path, "w", encoding="utf-8") as f:
            f.write(s)
    return df_info_path

def load_csv_info(path_cvs):
    cnt = 0
    df_info_path = read_df_info(path_cvs, over_read=False)  # 读取背景信息
    # test_data = io.StringIO(s)
    # df_2 = pd.read_csv(test_data,sep=" ")
    # print(df_2)
    name_types = {}  # 名称：类型
    type_ids = {}  # 类型：列名称
    type_names = {}  # 类型；名称
    name_list = []  # 方便索引建立
    with open(df_info_path, "r") as my_file:
        for line in my_file:
            cnt += 1
            if 5< cnt <= 87:
                this_line_list = line.split(' ')
                new_this_line_list = [i for i in this_line_list if i != '']
                this_id = int(new_this_line_list[0])
                name = new_this_line_list[1]
                this_type = new_this_line_list[4]
                if isinstance(this_id, int):
                    name_list.append(name)
                    name_types[name] = this_type  # name:type，对应。
                    if name.endswith('.Pv'):
                        this_value_type = 'sensor'
                        if this_value_type not in type_ids:
                            type_ids[this_value_type] =[this_id, ]
                            type_names[this_value_type] = [name, ]
                        else:
                            type_ids[this_value_type].append(this_id)
                            type_names[this_value_type].append(name)

                    elif name.endswith('Status') or '_STATE' in name:
                        this_value_type = 'actor'
                        if this_value_type not in type_ids:
                            type_ids[this_value_type] = [this_id, ]
                            type_names[this_value_type] = [name, ]
                        else:
                            type_ids[this_value_type].append(this_id)
                            type_names[this_value_type].append(name)
                    elif name.endswith('.Alarm'):
                        this_value_type = 'alarm'
                        if this_value_type not in type_ids:
                            type_ids[this_value_type] = [this_id, ]
                            type_names[this_value_type] = [name, ]
                        else:
                            type_ids[this_value_type].append(this_id)
                            type_names[this_value_type].append(name)
    return type_ids, type_names, name_types, name_list

def cal_useful_time_index(path_cvs, store_id_block_path):
    # 计算每个时间块对应的原始记录的id,并存储在store_id_block_path
    time_blocks = [['2019-12-06 10:05:00', '2019-12-06 10:21:00'],
                   ['2019-12-06 10:35:00', '2019-12-06 10:46:00'],
                   ['2019-12-06 10:50:00', '2019-12-06 11:01:00'],
                   ['2019-12-06 11:05:00', '2019-12-06 11:16:00'],
                   ['2019-12-06 11:20:00', '2019-12-06 11:31:00'],
                   ['2019-12-06 12:33:00', '2019-12-06 12:44:00'],
                   ['2019-12-06 12:46:00', '2019-12-06 12:57:00'],
                   ['2019-12-06 12:59:00', '2019-12-06 13:10:00'],
                   ['2019-12-06 13:12:00', '2019-12-06 13:23:00'],
                   ['2019-12-06 13:25:00', '2019-12-06 13:31:00'],
                   ['2019-12-06 13:31:01', '2019-12-06 13:46:00']
                   ]
    target_use_cols = [0, 1]
    df = pd.read_excel(path_cvs, skiprows=9, usecols=target_use_cols)  # 跳过前9行
    useful_time_index_dict = {}  # 存储时间片对应的id.
    # indicator = 0
    for i in df.index.values:
        row_data = df.loc[i].to_dict()  # 读取第几行
        target_time = row_data['t_stamp']
        target_time_str = target_time.strftime("%Y-%m-%d %H:%M:%S")#转换时间字符串
        for each_time_block in time_blocks:
            first_time_str = each_time_block[0]
            bool_flag = compute_remuneration(each_time_block, target_time_str)#计算在目标区间的id
            if bool_flag:
                if first_time_str not in useful_time_index_dict.keys():
                    useful_time_index_dict[first_time_str] = [i, ]
                    break
                else:
                    useful_time_index_dict[first_time_str].append(i)
                    break
            else:
                continue

    store_data(useful_time_index_dict, store_id_block_path)

def cal_abnormal_time_index(path_cvs, abnoraml_id_block_path):
    # 计算每个时间块对应的原始记录的id,并存储在store_id_block_path
    time_blocks = [['2019-12-06 10:20:00', '2019-12-06 10:30:00'],
                   ['2019-12-06 10:30:00', '2019-12-06 10:35:00'],
                   ['2019-12-06 10:45:00', '2019-12-06 10:50:00'],
                   ['2019-12-06 11:00:00', '2019-12-06 11:05:00'],
                   ['2019-12-06 11:15:00', '2019-12-06 11:20:00'],
                   ['2019-12-06 12:30:00', '2019-12-06 12:33:00'],
                   ['2019-12-06 12:43:00', '2019-12-06 12:46:00'],
                   ['2019-12-06 12:56:00', '2019-12-06 12:59:00'],
                   ['2019-12-06 13:09:00', '2019-12-06 13:12:00'],
                   ['2019-12-06 13:22:00', '2019-12-06 13:25:00']
                   ]
    target_use_cols = [0, 1]
    df = pd.read_excel(path_cvs, skiprows=9, usecols=target_use_cols)  # 跳过前9行
    abnormal_time_index_dict = {}  # 存储时间片对应的id.
    # indicator = 0
    for i in df.index.values:
        row_data = df.loc[i].to_dict()  # 读取第几行
        target_time = row_data['t_stamp']
        target_time_str = target_time.strftime("%Y-%m-%d %H:%M:%S")#转换时间字符串
        for each_time_block in time_blocks:
            first_time_str = each_time_block[0] #time_blocks对应的区间
            bool_flag = compute_remuneration(each_time_block, target_time_str)#计算在目标区间的id
            if bool_flag:
                if first_time_str not in abnormal_time_index_dict.keys():
                    abnormal_time_index_dict[first_time_str] = [i, ]
                    break
                else:
                    abnormal_time_index_dict[first_time_str].append(i)
                    break
            else:
                continue

    store_data(abnormal_time_index_dict, abnoraml_id_block_path)
    
def load_useful_id_time_blocks(store_id_block_path):
    # 读取cal_useful_time_index()计算的有效时间区间对应的id
    time_blocks_dict = read_pickle(store_id_block_path)
    # print(list(time_blocks.keys())[1:8])
    # some_time_blocks_keys = list(time_blocks_dict.keys())[1:9]  # 1：8，每个10min,661条记录。
    all_time_blocks_keys = list(time_blocks_dict.keys()) # 1：8，每个10min,661条记录。
    # print("all_time_blocks_keys:",all_time_blocks_keys)
    return all_time_blocks_keys, time_blocks_dict

def merge_dict(x:dict,y:dict):
    # 字典合并，相同k的v合并，不同添加
    for k ,v in x.items():
        if k in y.keys():
            y[k] = list(y[k]) + list(x[k]) # 相同k,则v合并
        else:
            y[k] = v # 若不同，则k 添加到y
    return y

def dict_slice(adict, end):
    # 字典切片，只有最后end位
    keys = adict.keys()
    dict_slice_data = {}
    for k in list(keys)[:end]:
        dict_slice_data[k] = adict[k]
    return dict_slice_data

def merge_each_time_block(some_time_blocks_keys, time_blocks_dict):
    #将time_blocks_dict合并成规整的list格式
    all_id_list = []
    all_start_id_list = [0,] # 记录每个开始id的开始
    #print("锚点个数：",len(some_time_blocks_keys),some_time_blocks_keys)
    for each_time_blocks in some_time_blocks_keys:
        this_block_ids = time_blocks_dict[each_time_blocks]  # 该区间对应的有效id
        all_start_id_list_cp = all_start_id_list.copy()
        all_start_id_list.append(len(this_block_ids[:600]) +  all_start_id_list_cp[-1])#人工校验后的逻辑id,用户画图时的x坐标。
        all_id_list.extend(this_block_ids[:600])
    #all_start_id_list.append(len())
    print("有效区间开始id：",all_start_id_list)

    return all_id_list,all_start_id_list

def merge_each_time_block_improved_mul(some_time_blocks_keys, time_blocks_dict):
    #将time_blocks_dict合并成规整的list格式
    all_id_list = []
    all_start_id_list = [0,] # 记录每个开始id的开始
    all_len = []
    all_start_time_str = []
    # print("锚点个数：",len(some_time_blocks_keys),some_time_blocks_keys)
    for each_time_blocks in some_time_blocks_keys:
        this_block_ids = time_blocks_dict[each_time_blocks]  # 该区间对应的有效id
        all_start_id_list_cp = all_start_id_list.copy()
        all_start_id_list.append(len(this_block_ids) +  all_start_id_list_cp[-1])#人工校验后的逻辑id,用户画图时的x坐标。
        all_id_list.extend(this_block_ids)
        all_len.append(len(this_block_ids))
        all_start_time_str.append(each_time_blocks)
    print("有效区间开始id：",all_start_id_list)
    print("all_len:",all_len)
    print("每个区间的开始时间：",all_start_time_str)
    return all_id_list, all_start_id_list

def merge_each_time_block_improved(some_time_blocks_keys, time_blocks_dict):
        # 将time_blocks_dict合并成规整的list格式
        all_id_list = []
        all_start_id_list = [0, ]  # 记录每个开始id的开始
        all_len = []
        all_start_time_str = []
        # print("锚点个数：",len(some_time_blocks_keys),some_time_blocks_keys)
        for each_time_blocks in some_time_blocks_keys:
            this_block_ids = time_blocks_dict[each_time_blocks]  # 该区间对应的有效id
            all_start_id_list_cp = all_start_id_list.copy()
            all_start_id_list.append(len(this_block_ids) + all_start_id_list_cp[-1])  # 人工校验后的逻辑id,用户画图时的x坐标。
            all_id_list.extend(this_block_ids)
            all_len.append(len(this_block_ids))
            all_start_time_str.append(each_time_blocks)
        print("有效区间开始id：", all_start_id_list)
        print("all_len:", all_len)
        print("每个区间的开始时间：", all_start_time_str)
        return all_id_list,all_start_id_list

def compute_remuneration(each_time_block, target_time_str):
    first_action_time = each_time_block[0]
    last_action_time = each_time_block[1]
    f1 = datetime.datetime.strptime(first_action_time, "%Y-%m-%d %H:%M:%S")
    f2 = datetime.datetime.strptime(last_action_time, "%Y-%m-%d %H:%M:%S")
    target_time = datetime.datetime.strptime(
        target_time_str, "%Y-%m-%d %H:%M:%S")
    if f1 <= target_time <= f2:
        return True
    else:
        return False

def get_used_time_index_0(path_cvs):
    time_blocks = [['2019-12-06 10:05:00', '2019-12-06 10:21:00'],
                   ['2019-12-06 10:35:00', '2019-12-06 10:46:00'],
                   ['2019-12-06 10:50:00', '2019-12-06 11:01:00'],
                   ['2019-12-06 11:05:00', '2019-12-06 11:16:00'],
                   ['2019-12-06 11:20:00', '2019-12-06 11:31:00'],
                   ['2019-12-06 12:33:00', '2019-12-06 12:44:00'],
                   ['2019-12-06 12:46:00', '2019-12-06 12:57:00'],
                   ['2019-12-06 12:59:00', '2019-12-06 13:10:00'],
                   ['2019-12-06 13:12:00', '2019-12-06 13:23:00'],
                   ['2019-12-06 13:25:00', '2019-12-06 13:31:00'],
                   ['2019-12-06 13:31:01', '2019-12-06 13:46:00']
                   ]
    target_use_cols = [0, 1]
    df = pd.read_excel(path_cvs, skiprows=9, usecols=target_use_cols)  # 跳过前9行
    # list(df.index) # 索引，
    # list(df.columns)#names
    # print(df.values)
    # print(list(df.index))
    useful_time_index_dict = {}  # 存储时间片对应的id.
    # indicator = 0
    for i in df.index.values:
        row_data = df.loc[i].to_dict()  # 读取第几行
        # print(type(row_data),row_data)
        target_time = row_data['t_stamp']
        target_time_str = target_time.strftime("%Y-%m-%d %H:%M:%S")
        print(i, target_time_str)  # 打印该值

        for each_time_block in time_blocks:
            index_now = time_blocks.index(each_time_block)
            first_time_str = each_time_block[0]
            bool_flag, int_flag = compute_remuneration(
                each_time_block, target_time_str)
            if bool_flag:
                if first_time_str not in useful_time_index_dict.keys():
                    useful_time_index_dict[first_time_str] = [i, ]
                    break
                else:
                    useful_time_index_dict[first_time_str].append(i)
                    break
            else:
                # 不在当前time_block,游走到下一个区间
                # indicator += 1
                # print("观测中间值：",time_blocks[indicator])
                if int_flag == 1:
                    # 说明比该区间的起始值还小，则后续区间不需要再比较
                    break
                elif int_flag == 2:
                    # 说明比该区间的终值大，则需开辟新的区间
                    indicator = index_now + 1
                    print("指示：", indicator)
                    each_time_block_next = time_blocks[indicator]
                    next_first_time_str = each_time_block[0]
                    bool_flag_2, int_flag_2 = compute_remuneration(
                        each_time_block_next, target_time_str)
                    if bool_flag_2:
                        useful_time_index_dict[next_first_time_str] = [
                            i, ]  # 直接创建新的key
                    else:
                        break  # 不能连跳区间，只能是递增的情况，因此，如果在下个新区间没有成功，则说明该值不需要处理，直接跳过。
    print(useful_time_index_dict)

def diff_smooth_0(fig_store_path_diff,ts,each_columns_name,cnt_fig,all_start_id_list,fig_x_items):

    '''时间序列平滑处理'''
    #wide = interval/60 # 间隔为1小时
    wide = 5 # 1秒为间隔
    # 差分序列
    dif = ts.diff().dropna()
    # 描述性统计得到：min，25%，50%，75%，max值
    td = dif.describe()
    # 定义高点阈值，1.5倍四分位距之外
    high = td['75%'] + 1.5 * (td['75%'] - td['25%'])
    # 定义低点阈值
    low = td['25%'] - 1.5 * (td['75%'] - td['25%'])

    i = 0
    forbid_index = dif[(dif > high) | (dif < low)].index
    while i < len(forbid_index) - 1:
        # 发现连续多少个点变化幅度过大
        n = 1
        # 异常点的起始索引
        start = forbid_index[i]
        if (i+n) < len(forbid_index)-1:
            while forbid_index[i+n] == start + datetime.timedelta(seconds=n):
                n += 1
        i += n - 1
        # 异常点的结束索引
        end = forbid_index[i]
        # 用前后值的中间值均匀填充
        value = np.linspace(ts[start - datetime.timedelta(seconds=wide)], ts[end + datetime.timedelta(seconds=wide)], n)
        ts[start: end] = value
        i += 1

    #return ts
    print("展示最新的数据格式：",ts)
    #all_data = ts.tolist()
    #N = len(all_data)
    fig, ax = plt.subplots()
    # xticks = all_start_id_list #this_block_ids
    ymin = np.min(ts)
    ymax = np.max(ts)
    # ax.set_xticks(xticks)
    ax.xaxis.set_major_locator(MultipleLocator(661))
    ax.plot(fig_x_items, ts)
    ax.vlines(all_start_id_list, ymin, ymax, linestyles='dashed', colors='red')
    cnt_fig += 1
    fig_name = str(cnt_fig) + '_' + each_columns_name + '.jpg'
    each_fig_path = fig_store_path_diff + '/' + fig_name
    plt.savefig(each_fig_path)
    # plt.show()
    plt.close()

def diff_smooth_fig(fig_store_path_diff,df,each_columns_name,cnt_fig,all_start_id_list,fig_x_items):

    '''时间序列平滑处理'''
    #wide = interval/60 # 间隔为1小时
    #wide = 5 # 1秒为间隔
    # 差分序列
    #dif = ts.diff().dropna()
    # dif = ts
    # # 描述性统计得到：min，25%，50%，75%，max值
    # td = dif.describe()
    # # 定义高点阈值，1.5倍四分位距之外
    # high = td['75%'] + 1.5 * (td['75%'] - td['25%'])
    # # 定义低点阈值
    # low = td['25%'] - 1.5 * (td['75%'] - td['25%'])
    #
    # i = 0
    # forbid_index = dif[(dif > high) | (dif < low)].index
    # while i < len(forbid_index) - 1:
    #     # 发现连续多少个点变化幅度过大
    #     n = 1
    #     # 异常点的起始索引
    #     start = forbid_index[i]
    #     if (i+n) < len(forbid_index)-1:
    #         while forbid_index[i+n] == start + datetime.timedelta(seconds=n):
    #             n += 1
    #     i += n - 1
    #     # 异常点的结束索引
    #     end = forbid_index[i]
    #     # 用前后值的中间值均匀填充
    #     value = np.linspace(ts[start - datetime.timedelta(seconds=wide)], ts[end + datetime.timedelta(seconds=wide)], n)
    #     ts[start: end] = value
    #     i += 1

    #return ts
    print("展示最新的数据格式：",df)
    decomposition = sm.tsa.seasonal_decompose(df,two_sided = False,model = 'additive',period=1)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    #----------------------------
    # all_data = ts.tolist()
    # N = len(all_data)
    # fig, ax = plt.subplots()
    # xticks = all_start_id_list #this_block_ids
    # ymin = np.min(df)
    # ymax = np.max(df)
    # ax.set_xticks(xticks)
    # ax.xaxis.set_major_locator(MultipleLocator(661))
    # ax.plot(fig_x_items, seasonal)
    # ax.vlines(all_start_id_list, ymin, ymax, linestyles='dashed', colors='red')
    # # cnt_fig += 1
    # fig_name = str(seasonal) + '_' + each_columns_name + '.jpg'
    # each_fig_path = fig_store_path_diff + '/' + fig_name
    # plt.savefig(each_fig_path)
    #plt.show()
    decomposition.plot()
    plt.show()
    # plt.close()

def diff_smooth(fig_store_path_diff,df,each_columns_name,cnt_fig,all_start_id_list,fig_x_items):

    fft_series = fft(df.values)
    power = np.abs(fft_series)
    sample_freq = fftfreq(fft_series.size)

    pos_mask = np.where(sample_freq > 0)
    freqs = sample_freq[pos_mask]
    powers = power[pos_mask]

    top_k_seasons =10
    # top K=3 index
    top_k_idxs = np.argpartition(powers, -top_k_seasons)[-top_k_seasons:]
    top_k_power = powers[top_k_idxs]
    fft_periods = (1 / freqs[top_k_idxs]).astype(int)

    print(f"top_k_power: {top_k_power}")
    print(f"fft_periods: {fft_periods}")

if __name__ == "__main__":
    pcap = "/home/ztf/Downloads/A6/pcap/Dec2019_00013_20191206131500.pcap"
    path_traffic = "/home/ztf/Downloads/A6/pcap"
    path_cvs = "/home/ztf/Downloads/A6/csv/Dec2019.xlsx"
    store_id_block_path = "useful_id_blocks"  # 存储计算好时间区间的id块。
    abnoraml_id_block_path = "abnormal_id_blocks"
    
    #cal_abnormal_time_index(path_cvs, abnoraml_id_block_path)
    # 只运行一次
    # cal_useful_time_index(path_cvs, store_id_block_path)
    # load_useful_id_time_blocks(store_id_block_path)
    all_time_blocks_keys, time_blocks_dict  = load_useful_id_time_blocks(abnoraml_id_block_path)
    print("RUN result:",all_time_blocks_keys)