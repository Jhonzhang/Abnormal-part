# -*- coding: UTF-8 -*-
import time
#from intervals import FloatInterval
import datetime
# import pyshark
import common_fun as cf
import math
import ipaddress

def now_time_str():
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    return local_time
    #print(local_time)
    

def strtime_to_timestamp(strtime):
    # 先转换为时间数组
    time_array = time.strptime(strtime, "%Y-%m-%d %H:%M:%S")

    # 转换为时间戳
    timestamp = time.mktime(time_array)#float类型
    #print(type(timeStamp))
    #print(timestamp)
    return int(timestamp)

def judge_timestamp_in_time_interval(test_timestamp,start_time,end_time):
    # intervals = FloatInterval(start_timestamp,end_timestamp)
    # print(intervals)
    # if test_timestamp  in intervals:
    #     print("ok")
    # else:
    #     print("不在该区间")
    #intervals = FloatInterval(start_timestamp,end_timestamp)
    #print(intervals)
    #开闭区间
    #print(type(test_time))
    #test_timestamp = strtime_to_timestamp(test_time)
    start_timestamp = strtime_to_timestamp(start_time)
    end_timestamp = strtime_to_timestamp(end_time)
    if test_timestamp >= start_timestamp and test_timestamp<end_timestamp:
        return True
        #print("ok")
    else:
        return False
        #print("不在该区间")
def format_time(the_starttime):
    #print("格式化前：",the_starttime)
    new_last_starttime = datetime.datetime.strptime(the_starttime, "%Y-%m-%d %H:%M:%S")
    new_last_starttime_timstamp = new_last_starttime.timestamp()
    str_t, datetime_1 = cf.timestamp_to_datatime3(new_last_starttime_timstamp)
    #datetime
    #print("格式化后：", str_t)
    return str_t,datetime_1#保留到毫秒级

def format_time2(the_starttime):
    #print("格式化前：",type(the_starttime),the_starttime)
    #new_last_starttime = datetime.datetime.strptime(the_starttime+".000", "%Y-%m-%d %H:%M:%S")
    #new_last_starttime_timstamp = new_last_starttime.timestamp()
    the_starttimeAarry = time.strptime(the_starttime,"%Y-%m-%d %H:%M:%S")
    the_starttimestamp = time.mktime(the_starttimeAarry)
    #str_time = datetime.datetime.strptime(the_starttime+".000000", "%Y-%m-%d %H:%M:%S.%f")
    str_t, datetime_1 = cf.timestamp_to_datatime3(the_starttimestamp)
    #datetime
    #print("格式化后：",str_t, datetime_1)
    return str_t,datetime_1#保留到毫秒级

def generate_last_action_endtime_new_method(last_action_starttime,step_end = 20):
    #默认在最后一个动作起始时间，往后扩展20s做为最后一个用户动作的结束时间
    #str3 = "2019-10-29 18:06:54"
    #pass
    
    new_last_starttime = datetime.datetime.strptime(last_action_starttime, "%Y-%m-%d %H:%M:%S")
    new_last_starttime_timstamp = new_last_starttime.timestamp()
    str_t,datetime_1 = cf.timestamp_to_datatime3(new_last_starttime_timstamp)
    #new_last_starttime = last_action_starttime.strftime("%Y-%m-%d %H:%M:%S")
    last_action_endtime = (datetime_1+datetime.timedelta(milliseconds=step_end)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    #print(type(last_action_endtime))
    return last_action_endtime

def generate_last_action_endtime_new_method2(last_action_starttime,step_end = 20):
    #默认在最后一个动作起始时间，往后扩展20s做为最后一个用户动作的结束时间
    #str3 = "2019-10-29 18:06:54"
    #pass
    
    new_last_starttime = datetime.datetime.strptime(last_action_starttime, "%Y-%m-%d %H:%M:%S.%f")
    new_last_starttime_timstamp = new_last_starttime.timestamp()
    str_t,datetime_1 = cf.timestamp_to_datatime3(new_last_starttime_timstamp)
    #new_last_starttime = last_action_starttime.strftime("%Y-%m-%d %H:%M:%S")
    last_action_endtime = (datetime_1+datetime.timedelta(milliseconds=step_end)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    #print(type(last_action_endtime))
    return last_action_endtime

def generate_last_action_endtime(last_action_starttime,step_end = 20):
    #默认在最后一个动作起始时间，往后扩展20s做为最后一个用户动作的结束时间
    #str3 = "2019-10-29 18:06:54"
    #pass
    print("run time compute")
    new_last_starttime = datetime.datetime.strptime(last_action_starttime, "%Y-%m-%d %H:%M:%S")
    #new_last_starttime = last_action_starttime.strftime("%Y-%m-%d %H:%M:%S")
    last_action_endtime = (new_last_starttime+datetime.timedelta(seconds=step_end)).strftime("%Y-%m-%d %H:%M:%S")
    #print(type(last_action_endtime))
    return last_action_endtime

def get_last_packet_times(pcap):

    cap = pyshark.FileCapture(pcap)
    #print(cap[0])
    cap.load_packets()
    #print(type(cap[0]))
    cnt = 0
    #print(len(cap))
    last_packet = cap[-1]
    #print(last_packet.sniff_time.strftime('%Y-%m-%d %H:%M:%S'))
    #print(last_packet.sniff_timestamp.split('.')[1])
    return last_packet.sniff_time.strftime('%Y-%m-%d %H:%M:%S'),int(last_packet.sniff_timestamp.split('.')[0])

def get_last_packet_times2(pcap):

    cap = pyshark.FileCapture(pcap)
    #print(cap[0])
    cap.load_packets()
    #print(type(cap[0]))
    cnt = 0
    #print(len(cap))
    last_packet = cap[-1]
    #print(last_packet.sniff_time.strftime('%Y-%m-%d %H:%M:%S'))
    #print(last_packet.sniff_timestamp.split('.')[1])
    return last_packet.sniff_time.strftime('%Y-%m-%d %H:%M:%S'),int(last_packet.sniff_timestamp.split('.')[0])

def timestamp_to_time(timstamp):
    #timstamp = 1611068750.543045
    time_local = time.localtime(timstamp)
    new_time = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
    #print(new_time)
    return new_time

def switch_relative_time_to_miliseconde(relative_time = 0):
    #relative_time_ms = 0

    global relative_time_str, relative_time_ms
    relative_time_str = ""
    relative_time_ms = ''
    relative_time = '{:.6f}'.format(relative_time)
    try:
        relative_time_str = str(relative_time)
        time_int_part = int(relative_time_str.split('.')[0])
        time_decimal_part = int(relative_time_str.split('.')[1])
        relative_time_ms = time_int_part*1000 + math.ceil(time_decimal_part/1000)
    except Exception as ex:
        print("错误类型：",ex,relative_time_str)
    # else:

    return relative_time_ms

def src_ip_pyshark(pcap):
    # timstamp_start = int(time.time())
    # print("开始时间：",timestamp_to_time(timstamp_start))
    cap = pyshark.FileCapture(pcap)
    #print(cap[0])
    cap.load_packets()
    #print(type(cap[0]))
    cnt = 0
    # timstamp_deal_start = int(time.time())
    # print("开始处理时间：", timestamp_to_time(timstamp_deal_start))
    # print(len(cap))
    last_packet = cap[-1]
    #the_last_action_start_time = cap[-1]
    # print(pcap)
    the_last_packet_time = last_packet.sniff_time.strftime('%Y-%m-%d %H:%M:%S')# 获取最后一个数据包的时间
    the_last_packet_timestamp = int(last_packet.sniff_timestamp.split('.')[0]) # 获取最后一个数据包的时间戳
    #if do_has_last_packet_time(pcap,last_action_time,the_last_action_end_timestamp):

    #print(last_packet)
    #pretty_print()
    src_ip_dict = {}
    dst_ip_dict = {}
    for each_packet in cap:
        #print(each_packet.ip.src)
        try:
            if each_packet.ip.src not in src_ip_dict.keys():
                src_ip_dict[each_packet.ip.src] = 1
            else:
                src_ip_dict[each_packet.ip.src] += 1
            if each_packet.ip.dst not in dst_ip_dict.keys():
                dst_ip_dict[each_packet.ip.dst] = 1
            else:
                dst_ip_dict[each_packet.ip.dst] += 1
        except:
            continue
    sort_src_ip = sorted(src_ip_dict, key=src_ip_dict.get, reverse=True)
    sort_dst_ip = sorted(dst_ip_dict, key=dst_ip_dict.get, reverse=True)
    max_src_ip = sort_src_ip[:3]
    max_dst_ip = sort_dst_ip[:3]
    cnt = 0
    #print("最大ip:",max_src_ip)
    back_ip = None
    if len(max_src_ip)!=0:
        #在判定其不为0，则进行后续的处理
        try:
            for each_ip in max_src_ip:
                if each_ip in max_dst_ip:
                    #print("简易判定方法：",each_ip)
                    if ipaddress.ip_address(each_ip.strip()).is_private:
                        back_ip = each_ip
        except Exception as e:
            print("ip判定错误：",e)
    #print("结束处理的时间：", timestamp_to_time(int(time.time())))
    return back_ip,the_last_packet_time,the_last_packet_timestamp #错误或为空值时，返回None

def formal_time_direct_output(time_str):
    # 直接输出
    return time_str

def formal_time(time_str):
    f_datetime = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    return f_datetime

def formal_time_0(time_str):

    f_datetime = time.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    timestamp = time.mktime(f_datetime)
    d = datetime.datetime.fromtimestamp(timestamp)
    time_stp = d.strftime("%Y-%m-%d %H:%M:%S")
    print('类型1：',type(time_str),type(time_stp))
    return time_stp

if __name__ == '__main__' :
    # str1 = "2019-10-29 18:06:50"
    # str2 = "2019-10-29 18:06:45"
    # str3 = "2019-10-29 18:06:54"
    # now_time_str()
    # the_starttime = "2021-06-17 10:35:00"
    pcap_4 = r'D:\dataset\applets_dataset\applets_dataset_7\songyi\songyi_mt_b1\traffic\mt_b1_30.pcap'
    back_content = src_ip_pyshark(pcap_4)
    if back_content is not None:
        print("检测到ip为：",back_content)


    # #the_last_packet_time, the_last_packet_timestamp = get_last_packet_times(pcap_4)
    # #print(the_last_packet_time, the_last_packet_timestamp)
    # format_time2(the_starttime)


    #generate_last_action_endtime(str3, step_end=20)

    #test_timestamp = strtime_to_timestamp(str1)
    # test_timestamp = 1572343615.855441000
    # start_timestamp = strtime_to_timestamp(str2)
    # end_timestamp = strtime_to_timestamp(str3)
    # judge_timestamp_in_time_interval(test_timestamp, start_timestamp, end_timestamp)


