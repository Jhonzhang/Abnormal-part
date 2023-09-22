import pickle

#from datetime import datetime
import datetime
#import joblib
#from sklearn.externals import joblib
import os 

def store_data(file, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(file, f,protocol=pickle.HIGHEST_PROTOCOL)

def read_pickle(file_name):
    f = open(file_name,'rb')
    content = pickle.load(f)
    f.close()
    return content


def store_data_big(file, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(file, f,protocol=pickle.HIGHEST_PROTOCOL)

def read_pickle_big(file_name):
    #f = open(file_name,'rb')
    if os.path.getsize(file_name) > 0 :
        with open(file_name ,"rb") as f:
            #content = pickle.load(f,fix_imports=True)
            unpickler = pickle.Unpickler(f)
            content = unpickler.load()
            #f.seek(0)
        #f.close()
        return content

def timestamp_to_datatime3(first_this_timestamp):
    # print(datetime.now())
    # t1 = timestamp.datetime.strftime("%Y-%m-%d %H:%M:%S.%f")[-3]
    # print("转换后的时间：",t1)
    #print("原始时间戳格式", first_this_timestamp)

    # t2 = time.localtime(stamp2)
    mm_timestamp = int(round(first_this_timestamp * 1000))  # 毫秒级时间戳
    # this_timestamp = time.localtime(mm_timestamp)
    #print("时间戳格式", mm_timestamp)
    timeStamp = float(mm_timestamp) / 1000
    d = datetime.datetime.utcfromtimestamp(timeStamp)
    d = d + datetime.timedelta(hours=8) # 中国默认时区
    str1 = d .strftime("%Y-%m-%d %H:%M:%S.%f")
    time1 = datetime.datetime.strptime(str1, "%Y-%m-%d %H:%M:%S.%f")
    #print("时间字符串：",str1[:-3])
    return str1[:-3],time1

def cal_time(stamp1,stamp2):

    t1, t11 = timestamp_to_datatime3(stamp1)
    t2, t22 = timestamp_to_datatime3(stamp2)
    delay = (t22 - t11)
    delay_second = delay.seconds  # 秒
    delay_microseconds = delay.microseconds  # 微妙
    all_milliseconds = int(delay_second * 1000 + delay_microseconds / 1000)  # 总毫秒
    #print("相差时间1:", delay_second)
    #print("相差时间1:", delay_microseconds)
    #print("相差时间2:", all_milliseconds)
    return all_milliseconds

def compute_remuneration_datetime_str(start_datetime_str,end_datetime_str):
    f1 = datetime.datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M:%S")
    f2 = datetime.datetime.strptime(end_datetime_str, "%Y-%m-%d %H:%M:%S")
    return (f2-f1).seconds


def time_diffence_datetime(first_this_timestamp,end_this_timstamp):

    t1, t11 = timestamp_to_datatime3(first_this_timestamp)
    t2, t22 = timestamp_to_datatime3(end_this_timstamp)
    delay  = (t22 - t11)
    delay_second = delay.seconds #秒
    delay_microseconds = delay.microseconds #微妙
    all_milliseconds = int(delay_second*1000+delay_microseconds/1000) #总毫秒
    #print("相差时间1:",delay_second)
    #print("相差时间1:",delay_microseconds)
    #print("相差时间2:",  all_milliseconds)
    return all_milliseconds

def foraml_time(time_str):
    #f_datetime = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    # f_datetime = time.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    # timestamp = time.mktime(f_datetime)
    # d = datetime.datetime.fromtimestamp(timestamp)
    # time_stp = d.strftime("%Y-%m-%d %H:%M:%S")
    # print('类型1：',type(time_str),type(time_stp))
    return time_str
if __name__ == "__main__":

    first_this_timestamp = 1436428275.207596
    end_this_timstamp =    1436428278.716832
    ztf = [0,1,2,3,'4',100]
    # print(namestr_fun(ztf,locals()))


    #print(time_diffence_datetime(first_this_timestamp, end_this_timstamp))

