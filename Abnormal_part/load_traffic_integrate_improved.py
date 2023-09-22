import dpkt
import pyshark
import time
import numpy as np
import pandas as pd
import statistics as st
import scipy.stats as ss
import math
from subprocess import Popen, PIPE
from common_fun import *
from common_time import generate_last_action_endtime, now_time_str, formal_time_direct_output
from multiprocessing import Pool

def deal_packet_pyshark(pcap_path, MAX_NUM=0):
    """
    处理每个pcap
    Args:
        pcap_path:数据包的绝对路径
        MAX_NUM: 测试时，设置一个值，这样就不需要运行全部数据，较为费时间。
    Returns:all_packets_timestamp,all_packets_len。均为list格式。

    """
    print("deal_packet:",now_time_str())
    # keep_packets: 设定在调用next()函数之后是否保留之前读取的数据包。用于在读取较大的捕获时节省内存。
    rule_bad_tcp_all = "!(tcp.analysis.flags && !tcp.analysis.window_update && !tcp.analysis.keep_alive && !tcp.analysis.keep_alive_ack)"
    rule_tcp_wrong_some = "!tcp.analysis.retransmission and !tcp.analysis.duplicate_ack and !tcp.analysis.ack_lost_segment and " \
                          "!tcp.analysis.spurious_retransmission and !tcp.analysis.out_of_order"
    cap = pyshark.FileCapture(pcap_path, keep_packets=False, display_filter=
    " !icmp and !ntp and !ssdp and !arp and " + rule_bad_tcp_all)  # cap.load_packets()#不要加载，循环遍历，速度一样很快！
    cnt = 0  # 数据包计数
    all_packets_timestamp = []
    all_packets_len = []
    bad_pkt_list = []
    for pkt in cap:
        try:
            if MAX_NUM != 0:

                if cnt > MAX_NUM:
                    break
            all_packets_timestamp.append(pkt.sniff_time)  # datetime.datetime(2019, 12, 6, 10, 5, 5, 370982)
            all_packets_len.append(int(pkt.length))  # frame长度
            cnt += 1
        except Exception as e:
            print("错误类型:", e)
            bad_pkt_list.append(pkt.sniff_time)
    cap.close()
    return all_packets_timestamp, all_packets_len


# 单独处理
def compute_remuneration(each_time_block, target_time_str):
    # 判定target_time_str 是否在each_time_block 时间区间内，返回bool值。
    first_action_time = each_time_block[0]
    last_action_time = each_time_block[1]
    f1 = datetime.datetime.strptime(first_action_time, "%Y-%m-%d %H:%M:%S")
    f2 = datetime.datetime.strptime(last_action_time, "%Y-%m-%d %H:%M:%S")
    target_time = datetime.datetime.strptime(target_time_str, "%Y-%m-%d %H:%M:%S")
    if f1 <= target_time <= f2:
        return True
    else:
        return False


def cal_used_time_index(path_cvs, stort_id_block_path):
    # 11个时间区间对应的原始id,dic存储。
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
    for i in df.index.values:
        row_data = df.loc[i].to_dict()  # 读取第几行
        target_time = row_data['t_stamp']
        target_time_str = target_time.strftime("%Y-%m-%d %H:%M:%S")
        for each_time_block in time_blocks:
            first_time_str = each_time_block[0]
            bool_flag = compute_remuneration(each_time_block, target_time_str)  # 判定是否在该时间区间
            if bool_flag:
                if first_time_str not in useful_time_index_dict.keys():
                    useful_time_index_dict[first_time_str] = [i, ]
                    break
                else:
                    useful_time_index_dict[first_time_str].append(i)
                    break
            else:
                continue
    store_data(useful_time_index_dict, stort_id_block_path)


# step 1:cut_pcap
def load_pcap_name_orgi(path_pcap):
    # 获取路径：/home/ztf/Downloads/A6/pcap/ 下的pcap文件名
    pcap_names = os.listdir(path_pcap)
    target_pcap_name_list = []
    for each_pcap_name in pcap_names:
        if '.pcap' in each_pcap_name:
            target_pcap_name_list.append(each_pcap_name)
    target_pcap_name_list.sort()  # 升序排列文件名
    return target_pcap_name_list


def cut_special_pcap(each_special_pcap_1, each_special_pcap_2, each_time_blocks, pcap_new_store_path):
    # 处理跨时间区间的双pcap,切分后返回sub_pcap绝对路径
    new_pcap = each_time_blocks[0].replace(' ', '_')
    new_pcap = new_pcap.replace('-', '_').replace(':', '_')
    pcap_new_path = os.path.join(pcap_new_store_path, new_pcap + '.pcap')
    start_each_time_blocks = formal_time_direct_output(each_time_blocks[0])
    end_each_time_blocks = formal_time_direct_output(each_time_blocks[1])
    merged_pcap = os.path.join(pcap_new_store_path, new_pcap + '_tmp_megred.pcap')  # 合并后的大数据包
    cmd_merge = r'mergecap -w %s %s %s' % (merged_pcap, each_special_pcap_1, each_special_pcap_2)
    ps_merge = Popen(cmd_merge, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    out = ps_merge.stdout
    return_code = ps_merge.wait()
    # time.sleep(3)
    print("状态码：", return_code)
    cmd = r'editcap -A "%s" -B "%s" %s %s' % (start_each_time_blocks, end_each_time_blocks, merged_pcap, pcap_new_path)
    ps = Popen(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    out2 = ps.stdout
    return pcap_new_path


def cut_common_pcap(each_common_pcap, each_time_blocks, pcap_new_store_path):
    # 切分正常的大pcap，返回sub_pcap 绝对路径
    new_pcap = each_time_blocks[0].replace(' ', '_')
    new_pcap = new_pcap.replace('-', '_').replace(':', '_')
    pcap_new_path = os.path.join(pcap_new_store_path, new_pcap + '.pcap')  # 返回sub_pcap绝对路径
    start_each_time_blocks = formal_time_direct_output(each_time_blocks[0])
    end_each_time_blocks = formal_time_direct_output(each_time_blocks[1])
    cmd = r'editcap -A "%s" -B "%s" %s %s' % (
    start_each_time_blocks, end_each_time_blocks, each_common_pcap, pcap_new_path)
    ps = Popen(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    out = ps.stdout
    return pcap_new_path


def cut_pcap(path_pcap):
    """
    根据Swat数据集记录的time_blocks和每个大pcap对应的时间区间，确定生成新的sub_pcap
    Args:
        path_pcap: /home/ztf/Downloads/A6/pcap/ 下的pcap文件名

    Returns:存储sub_pcap路径的dict,其key为以每个时间区间的开始时刻，value：对应的sub_pcap路径
    """
    target_pcap_name_list = load_pcap_name_orgi(path_pcap)  # 获取原始大pcap名称
    # Swat数据集记录的正常操作时间区间，即没有攻击的数据。
    time_blocks_normal = [['2019-12-06 10:05:00', '2019-12-06 10:21:00'],
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
    
    time_blocks_abnormal = [['2019-12-06 10:20:00', '2019-12-06 10:30:00'],
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
    # 手动切割,因大pcap的保存是以15min为单位，使得Swat正常操作的时间区间会存在跨两个大的pcap,因此第一步需要按记录的正常操作的时间区间
    # 切分出新的pcap（生成与时间区间对应的sub_pcap），然后按1s(sensor,actor以1s为单位抽样记录)切分sub_pcap,生成最小处理单元meta_pcap.
    store_time_pcap_name_dict = {}
    # 根据Swat数据集记录的time_blocks和每个大pcap对应的时间区间，确定生成新的sub_pcap对应的原始大pcap。
    target_pcap_name_normal = [target_pcap_name_list[0], target_pcap_name_list[2], target_pcap_name_list[3],
                        target_pcap_name_list[4], target_pcap_name_list[5], target_pcap_name_list[10],
                        target_pcap_name_list[11],
                        (target_pcap_name_list[11], target_pcap_name_list[12]),
                        (target_pcap_name_list[12], target_pcap_name_list[13]),
                        target_pcap_name_list[13], target_pcap_name_list[14]]
    
    target_pcap_name_abnormal = [target_pcap_name_list[1], target_pcap_name_list[2], target_pcap_name_list[3],
                        target_pcap_name_list[4], target_pcap_name_list[5], target_pcap_name_list[10],
                        (target_pcap_name_list[10], target_pcap_name_list[11]),
                        target_pcap_name_list[11],
                        target_pcap_name_list[12], target_pcap_name_list[13]]
    pcap_new_store_path_normal = '/home/ztf/Downloads/A6/pcap/pcap_new_1'  # sub_pcap存储路径
    pcap_new_store_path_abnormal = '/home/ztf/Downloads/A6/pcap/pcap_new_abnormal'  # sub_pcap存储路径
    if not os.path.exists(pcap_new_store_path_abnormal):
        os.makedirs(pcap_new_store_path_abnormal)
    for each_pcap_name_ori in target_pcap_name_abnormal:
        index_pcap = target_pcap_name_abnormal.index(each_pcap_name_ori)  # 获取当前的索引
        # 判定是否为跨时间区间的特殊大pcap对
        if isinstance(each_pcap_name_ori, tuple):
            each_special_pcap = path_pcap + '/' + each_pcap_name_ori[0]
            each_special_pcap_2 = path_pcap + '/' + each_pcap_name_ori[1]
            new_pcap_name_path = cut_special_pcap(each_special_pcap, each_special_pcap_2, time_blocks_abnormal[index_pcap],
                                                  pcap_new_store_path_abnormal)
        else:
            # 切分正常的大pcap
            each_common_pcap = path_pcap + '/' + each_pcap_name_ori
            new_pcap_name_path = cut_common_pcap(each_common_pcap, time_blocks_abnormal[index_pcap], pcap_new_store_path_abnormal)

        store_time_pcap_name_dict[time_blocks_abnormal[index_pcap][0]] = new_pcap_name_path  # 以每个时间区间的开始时刻为key值
    print("切分好的数据包", store_time_pcap_name_dict)
    store_dict_path_normal = 'project_ztf/store_time_pcap_name_dict'
    store_dict_path_abnormal = '/home/ztf/Abnormal_part/store_time_pcap_name_dict_abnormal'
    store_data(store_time_pcap_name_dict,store_dict_path_abnormal)


# step 2:cut_time_block_pcap
def cut_each_dealt_pcap(start_time_str, each_common_pcap, store_path):
    """将每个时间块内的数据包，按时间窗或数据个数块进行切割。
    以1s为单位时间块切分pcap后返回project_ztf/store_time_pcap_name_dict
    Args:
        start_time_str (_type_): _description_
        each_common_pcap (_type_): _description_
        store_path (_type_): _description_
    """
    new_pcap = start_time_str.replace(' ', '_')
    new_pcap = new_pcap.replace('-', '_').replace(':', '_')
    new_pcap_file_path = os.path.join(store_path, new_pcap)  # 存储每个个时间区间的meta_pcap文件夹
    if not os.path.exists(new_pcap_file_path):
        os.makedirs(new_pcap_file_path)
    first_new_pcap = os.path.join(new_pcap_file_path, new_pcap + '_new.pcap')
    i_time = 1  # 按1s为单位切分
    cmd_time = r'editcap -i %s %s %s ' % (i_time, each_common_pcap, first_new_pcap)  # 按时间切分的tshark指令特性
    ps = Popen(cmd_time, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    out = ps.stdout


def cut_time_block_pcap(store_dict_path_abnormal,store_path_abnormal):
    # 切分sub_pcap,生成meta_pcap，存储在pcap_after_dealed
    time_block_pcap_name_path_dict = read_pickle(store_dict_path_abnormal)  # 存储sub_pcap路径的dict,由cut_pcap函数生成
    pcap_path_list = list(time_block_pcap_name_path_dict.values())  # 所有的目标路径
    time_start = list(time_block_pcap_name_path_dict.keys())
    num_cores = len(pcap_path_list)  # 多进程处理时，创建核的个数
    
    if not os.path.exists(store_path_abnormal):
        os.makedirs(store_path_abnormal)
    # 不能多进程，太耗内存
    for i in range(num_cores):
        print("run", i)
        start_time_str = time_start[i]
        each_common_pcap = pcap_path_list[i]
        cut_each_dealt_pcap(start_time_str, each_common_pcap, store_path_abnormal)  # 切分meta_pcap

# step 3: improved
def run_each_sub_pcap(i, store_all_feature_path, this_file_path):
    #this_file_path：每个时间区间的sub_pcap路径
    time_features_dict = {}  # 每个after_dealt文件夹下，重建新存储空间，
    store_feature_file_name = "time_feature_all" + '_' + str(i)
    store_feature_file_name_path = os.path.join(store_all_feature_path, store_feature_file_name)#存储每个时间区间对应的特征的路径
    this_file_pcaps = os.listdir(this_file_path)  # 获取该文件下的文件名
    print("该文件夹的pcap个数:",len(sorted(this_file_pcaps)),store_feature_file_name_path)
    for each_pcap in sorted(this_file_pcaps):
        this_pcap_path = os.path.join(this_file_path, each_pcap)  # 每个meta_pcap数据包加载的路径，1s
        print("this pcap len:",this_pcap_path)
        all_packets_timestamp, all_packets_len = deal_packet_pyshark(this_pcap_path)  # 对每个pcap，不用并行。
        #all_packets_len = [1,2,3]
        print("观测",all_packets_len)
        time_features_dict[this_pcap_path] = all_packets_len
    store_data(time_features_dict, store_feature_file_name_path)

def cal_time_features_all(store_path,store_all_feature_path):
    # 与cal_time_features()不同有两点：1、该函数并行，2、存储的内容是时间区间的每个meta_pcap的数据包长度序列，后续仍需s
    # 运行statistic_feature()
    
    if not os.path.exists(store_all_feature_path):
        os.makedirs(store_all_feature_path)
    all_file_names = sorted(os.listdir(store_path))  # 文件夹个数
    print("总共要处理的文件夹：",len(all_file_names),all_file_names)
    num_cores = len(all_file_names)  # 创建核的个数
    p = Pool(num_cores)
    for i in range(num_cores):
        each_file_name = all_file_names[i]
        
        this_file_path = os.path.join(store_path, each_file_name)  # 每个时间块的sub_pcap的路径
        print(i, "当前处理的数据包：", this_file_path, now_time_str())
        p.apply_async(run_each_sub_pcap, args=(i, store_all_feature_path, this_file_path))#多进程处理，run_each_sub_pcap()
    p.close()
    p.join()

def merge_dict_improved(x: dict, y: dict):
    # 字典合并，相同k的v合并，不同添加
    for k, v in x.items():
        k_list = k.split('/')
        this_real_key = k_list[7] #获取的开始时间key
        # print(this_real_key,len(v))
        #print("this_real_key:",this_real_key)
        if this_real_key not in list(y.keys()):
            y[this_real_key] = [v, ]  # 若不同，则k 添加到y
        else:
            #print("观测值2", y.keys(),len(y.values()))
            y[this_real_key].append(v)  # 相同k,则v合并

    return y

### all_feature ### 如果时improved版本的代码，则不需要运行后续内容。
def load_traffic_scattered_feature_improve():
    # 加载cal_time_features_all生成的时间区间对应的packet length sequence
    scattered_feature_path = '/home/ztf/project_ztf/all_feature' #only len sequence
    all_file_names =sorted(os.listdir(scattered_feature_path))
    # print(len(all_file_names),"all_file_names:", all_file_names[:])
    # print(type(all_file_names[-1]))
    # new_all_file_name = all_file_names[1:3] + [all_file_names[-1]] + all_file_names[3:-3]# 裁剪出目标时间区间
    new_all_file_name = all_file_names
    print(len(new_all_file_name), "new_all_file_name:", new_all_file_name[:])
    all_dict = {}
    for each_file_name in new_all_file_name:
        each_file_path = os.path.join(scattered_feature_path, each_file_name)
        each_data = read_pickle(each_file_path)
        each_data_keys = list(each_data.keys())
        print(each_file_name,", each_block 包含数据量：",len(each_data_keys),each_data_keys[0])
        all_dict = merge_dict_improved(each_data, all_dict)
    print("all_dict:",len(all_dict.values()),list(all_dict.keys()))
    # list_tuple_dict = sorted(all_dict.items(), key=lambda d: d[0], reverse=False)
    # all_dict = dict(list_tuple_dict[:-1])  # 1:8 的字典，需要合并数据。
    # # print("The last result:",list(all_dict.keys()))
    all_traffic_feature_list = []
    for k, v in all_dict.items():
        this_block_data = v  # v[this_pcap_path] = statistic_feature(all_packets_len)
        # tmp_feature = []
        print("each_block_traffic_len:",len(this_block_data))
        for each_pcap_len_list in this_block_data:
        # for each_pcap_len_list in this_block_data[:600]:
            all_traffic_feature_list.append(statistic_feature(each_pcap_len_list))
        # all_traffic_feature_list.extend()  # 10min

    return all_traffic_feature_list
    # return pd.DataFrame(all_traffic_feature_list)

def load_traffic_scattered_feature_improved_mul():
    # 加载cal_time_features_all生成的时间区间对应的packet length sequence
    scattered_feature_path = '/home/ztf/project_ztf/all_feature/all_feature_improve' #only len sequence
    all_file_names =sorted(os.listdir(scattered_feature_path))
    # print(len(all_file_names),"all_file_names:", all_file_names[:])
    # print(type(all_file_names[-1]))
    # new_all_file_name = all_file_names[1:3] + [all_file_names[-1]] + all_file_names[3:-3]# 裁剪出目标时间区间
    new_all_file_name = all_file_names
    # print(len(new_all_file_name), "new_all_file_name:", new_all_file_name[:])
    all_dict = {}
    for each_file_name in new_all_file_name:
        each_file_path = os.path.join(scattered_feature_path, each_file_name)
        each_data = read_pickle(each_file_path)
        each_data_keys = list(each_data.keys())
        # print(each_file_name,", each_block 包含数据量：",len(each_data_keys),each_data_keys[0])
        all_dict = merge_dict_improved(each_data, all_dict)
    list_tuple_dict = sorted(all_dict.items(), key=lambda d: d[0], reverse=False)# 字符串升序排列。
    all_dict = dict(list_tuple_dict)
    print("all_dict:",len(all_dict.values()),list(all_dict.keys()))
    # list_tuple_dict = sorted(all_dict.items(), key=lambda d: d[0], reverse=False)
    # all_dict = dict(list_tuple_dict[:-1])  # 1:8 的字典，需要合并数据。
    # # print("The last result:",list(all_dict.keys()))
    all_traffic_feature_list = []
    all_block_len_list = []
    for k, v in all_dict.items():
        this_block_data = v  # v[this_pcap_path] = statistic_feature(all_packets_len)
        # tmp_feature = []
        print("each_block_traffic_len:",len(this_block_data))
        all_block_len_list.append(len(this_block_data))
        this_block_feature = []
        for each_pcap_len_list in this_block_data:
        # for each_pcap_len_list in this_block_data[:600]:
            this_block_feature.append(statistic_feature(each_pcap_len_list))

        all_traffic_feature_list.append(this_block_feature) #每个区间的数据在处理后都添加到总的list中。
    all_block_len_list[-1] = 840
    all_traffic_feature_list[-1] = all_traffic_feature_list[-1][0:840]
    return all_traffic_feature_list,all_block_len_list
    # return pd.DataFrame(all_traffic_feature_list)

def load_traffic_scattered_feature_improved_mul_abnormal(scattered_feature_path):
    # 加载cal_time_features_all生成的时间区间对应的packet length sequence
    #cattered_feature_path = '/home/ztf/project_ztf/all_feature' #only len sequence
    all_file_names =sorted(os.listdir(scattered_feature_path))
    # print(len(all_file_names),"all_file_names:", all_file_names[:])
    # print(type(all_file_names[-1]))
    # new_all_file_name = all_file_names[1:3] + [all_file_names[-1]] + all_file_names[3:-3]# 裁剪出目标时间区间
    new_all_file_name = all_file_names
    # print(len(new_all_file_name), "new_all_file_name:", new_all_file_name[:])
    all_dict = {}
    for each_file_name in new_all_file_name:
        # print("each_pcap_name:",each_file_name)
        each_file_path = os.path.join(scattered_feature_path, each_file_name)
        each_data = read_pickle(each_file_path)
        each_data_keys = list(each_data.keys())
        # print(each_file_name,", each_block 包含数据量：",len(each_data_keys),each_data_keys[0],each_data_keys[1])
        all_dict = merge_dict_improved(each_data, all_dict)
    list_tuple_dict = sorted(all_dict.items(), key=lambda d: d[0], reverse=False)# 字符串升序排列。
    all_dict = dict(list_tuple_dict)
    # print("all_dict:",len(all_dict.values()),list(all_dict.keys()))
    # list_tuple_dict = sorted(all_dict.items(), key=lambda d: d[0], reverse=False)
    # all_dict = dict(list_tuple_dict[:-1])  # 1:8 的字典，需要合并数据。
    # # print("The last result:",list(all_dict.keys()))
    all_traffic_feature_list = []
    all_block_len_list = []
    for k, v in all_dict.items():
        this_block_data = v  # v[this_pcap_path] = statistic_feature(all_packets_len)
        # tmp_feature = []
        # print("each_block_traffic_len:",len(this_block_data))
        all_block_len_list.append(len(this_block_data))
        this_block_feature = []
        for each_pcap_len_list in this_block_data:
        # for each_pcap_len_list in this_block_data[:600]:
            this_block_feature.append(statistic_feature(each_pcap_len_list))

        all_traffic_feature_list.append(this_block_feature) #每个区间的数据在处理后都添加到总的list中。
    # all_block_len_list[-1] = 840
    # all_traffic_feature_list[-1] = all_traffic_feature_list[-1][0:840]
    return all_traffic_feature_list[1:],all_block_len_list[1:]
    # return pd.DataFrame(all_traffic_feature_list)
###############################################
# step 3:cal_time_features,提取特征,费improved版本的代码，才需要运行后续内容。
def cal_variance(pkt_len_list):
    # 返回前后四分位
    first_quartile_index = math.ceil(len(pkt_len_list) / 4) - 1
    second_quartile_index = math.ceil(len(pkt_len_list) / 2) - 1
    third_quartile_index = math.ceil(len(pkt_len_list) * 3 / 4) - 1
    first_quartile_front_array = np.array(pkt_len_list[:first_quartile_index])
    first_quartile_back_array = np.array(pkt_len_list[first_quartile_index + 1:])
    second_quartile_front_array = np.array(pkt_len_list[:second_quartile_index])
    second_quartile_back_array = np.array(pkt_len_list[second_quartile_index + 1:])
    third_quartile_front_array = np.array(pkt_len_list[:third_quartile_index])
    third_quartile_back_array = np.array(pkt_len_list[third_quartile_index + 1:])
    return [round(np.var(first_quartile_front_array), 3), round(np.var(first_quartile_back_array), 3),
            round(np.var(second_quartile_front_array), 3),
            round(np.var(second_quartile_back_array), 3), round(np.var(third_quartile_front_array), 3),
            round(np.var(third_quartile_back_array), 3)]

def statistic_feature(pkt_len_list):
    # 返回统计特征值
    size_db_array = np.array(pkt_len_list)
    # print("view:",size_db_array)
    mean_size_db = round(ss.tmean(size_db_array), 3)
    std_size_db = round(ss.tstd(size_db_array), 3)
    kurtosis_size_db = round(ss.kurtosis(size_db_array), 3)
    skew_size_db = round(ss.skew(size_db_array), 3)
    min_db = np.min(size_db_array)
    max_db = np.max(size_db_array)
    difference_value_db = np.round((np.sum(size_db_array) / len(pkt_len_list)), 3)  # 平均字节数
    mode_db = st.mode(size_db_array)
    tvar_db = round(ss.tvar(size_db_array), 3)
    variance_db = cal_variance(pkt_len_list)  # 返回6个四分位值
    # print("variance:",variance_db) #.extend(variance_db)
    return [mean_size_db, std_size_db, kurtosis_size_db, skew_size_db, min_db, max_db, difference_value_db, mode_db,
            tvar_db] + variance_db

def cal_time_features():
    """
    如果当初不是手动运行多个文件（第一版本代码是的多进程总是错误！转而使用多脚本分批次运行的笨方法），造成all_file_names索引错误，也不会有后续这么麻烦事情！！！
    现在的程序是修改后的最终版本。
    Returns:
    """
    time_features_dict = {}  # 时间与特征值的索引
    store_path = '/home/ztf/Downloads/A6/pcap/pcap_after_dealt'
    store_all_feature_path = '/home/ztf/project_ztf/network_traffic_scattered_feature'
    if not os.path.exists(store_all_feature_path):
        os.makedirs(store_all_feature_path)
    all_file_names = sorted(os.listdir(store_path))  # 包含了所有原始时间块（官方文档定义的时间）对应的数据包。
    # for 对应的all_file_names 在第一版本的代码是局部元素的多脚本load_traffic_x.py运行。现在该为规整代码为如下内容，在第二版本中解决了多进程的问题。
    # for each_file_name in [all_file_names[0],all_file_names[1],all_file_names[5]):
    for each_file_name in all_file_names:
        print("当前处理的数据包：", each_file_name, now_time_str())
        time_features_dict[each_file_name] = {}  # 对每个时间快的数据包进行处理后存储。
        this_time_block_index = all_file_names.index(each_file_name)
        store_feature_file_name = "time_feature" + "_" + str(this_time_block_index)
        store_feature_file_name_path = os.path.join(store_all_feature_path, store_feature_file_name)  # 每个时间块的特征
        this_file_path = os.path.join(store_path, each_file_name)  # 每个时间块的meta_pcap
        this_file_pcaps = os.listdir(this_file_path)
        for each_pcap in sorted(this_file_pcaps):
            this_pcap_path = os.path.join(this_file_path, each_pcap)  # 每个数据包加载的路径，1s
            all_packets_timestamp, all_packets_len = deal_packet_pyshark(this_pcap_path)  # 提取时间和长度序列。
            time_features_dict[each_file_name][this_pcap_path] = statistic_feature(all_packets_len)
        store_data(time_features_dict, store_feature_file_name_path)

# step 4:对齐特征
def merge_dict(x: dict, y: dict):
    # 字典合并，相同k的v合并，不同添加
    for k, v in x.items():
        if k in y.keys():
            y[k] = list(y[k]) + list(x[k])  # 相同k,则v合并
        else:
            y[k] = v  # 若不同，则k 添加到y
    return y

# 对齐10min
def load_traffic_scattered_feature(scattered_feature_path):
    """
    Args:
        scattered_feature_path:已经提取的每个时间块的流量特征

    Returns: 整合后的流量特征list.

    """
    all_file_names = os.listdir(scattered_feature_path)
    all_dict = {}
    for each_file_name in all_file_names:
        each_file_path = os.path.join(scattered_feature_path, each_file_name)
        each_data = read_pickle(each_file_path)
        # print(each_file_name,"each_block 包含数据量：",list(each_data.keys()))
        all_dict = merge_dict(each_data, all_dict)
    # print("all_dict:",all_dict.keys())
    list_tuple_dict = sorted(all_dict.items(), key=lambda d: d[0], reverse=False)
    all_dict = dict(list_tuple_dict[:-1])  # 1:8 的字典，需要合并数据。
    # print("The last result:",list(all_dict.keys()))
    all_traffic_feature_list = []
    for k, v in all_dict.items():
        this_block_data = list(v.values())  # v[this_pcap_path] = statistic_feature(all_packets_len)
        all_traffic_feature_list.extend(this_block_data[:600])  # 10min
    return all_traffic_feature_list
    # return pd.DataFrame(all_traffic_feature_list)

# 在本文件中的测试
def main():
    """
    测试load_traffic_scattered_feature函数
    Returns:

    """
    scattered_feature_path = 'all_feature/network_traffic_scattered_feature'  # 提取的流量特征。
    all_traffic_feature_list = load_traffic_scattered_feature(scattered_feature_path)
    print("打印测试结果", type(all_traffic_feature_list), len(all_traffic_feature_list),
          len(all_traffic_feature_list[0]))


if __name__ == "__main__":
    pcap = "Downloads/A6/pcap/Dec2019_00000_20191206100500.pcap"
    path_pcap = "/home/ztf/Downloads/A6/pcap"
    path_cvs = "/home/ztf/Downloads/A6/csv/Dec2019.xlsx"
    pcap_f = '/home/ztf/Downloads/A6/pcap/pcap_new_1/2019_12_06_10_05_00.pcap'
    store_name = '/home/ztf/Downloads/A6/pcap/pcap_after_dealt'
    store_dict_path_normal = 'project_ztf/store_time_pcap_name_dict'
    store_dict_path_abnormal = '/home/ztf/Abnormal_part/store_time_pcap_name_dict_abnormal'
    store_path_normal = '/home/ztf/Downloads/A6/pcap/pcap_after_dealt'  # 存储meta_pcap
    store_path_abnormal = '/home/ztf/Downloads/A6/pcap/pcap_after_dealt_abnormal'  # 存储异常meta_pcap
    
    # step 1：提取每个数据包的time,len 序列。只运行一次，把提取的结果存储起来。
    # cut_pcap(path_pcap) #产生store_time_pcap_name_dict

    # step 2:#切分sub_pcap to meta_pcap,只运行一次，把提取的结果存储起来。
    # cut_time_block_pcap(store_dict_path_abnormal,store_path_abnormal)
    
    # step 3：切分出长度序列，运行8个小时。！！！！太慢了！，只运行一次
    #store_path = '/home/ztf/Downloads/A6/pcap/pcap_after_dealt' #按时间区间的meta_pcap
    #store_all_feature_path = '/home/ztf/project_ztf/all_feature' #存储切分好的特征存放路径,每个meta_pcap的数据包长度序列。
    store_all_feature_path_abnormal = '/home/ztf/Abnormal_part/all_feature_abnormal' #存储切分好的特征存放路径,每个meta_pcap的数据包长度序列。
    #cal_time_features(store_path_abnormal,store_all_feature_path_abnormal) # 处理正常的数据块,只运行一次，把提取的结果存储起来。
    #cal_time_features_all(store_path_abnormal,store_all_feature_path_abnormal)

    # step 4:返回对齐后的traffic feature
    # load_traffic_scattered_feature()

    # main()#测试load_traffic_scattered_feature函数
    # useful_id = read_pickle('useful_id_blocks')
    # print(len(useful_id), useful_id)

    # all_traffic_feature_list,all_block_len_list = load_traffic_scattered_feature_improved_mul()
    # print(len(all_traffic_feature_list))
    # print("all_block_len_list:",all_block_len_list)
    
    all_traffic_feature_list,all_block_len_list = load_traffic_scattered_feature_improved_mul_abnormal(store_all_feature_path_abnormal)
    print(len(all_traffic_feature_list))
    print("all_block_len_list:",all_block_len_list)
