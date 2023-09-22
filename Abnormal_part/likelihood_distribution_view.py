import numpy as np
from common_time import now_time_str
from common_fun import *
from collections import Counter
np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('agg')
import statistics
import seaborn as sns

def cal_abnormal_sts():
    finename = "likelihood.txt"
    cnt = 0
    all_abnormal_likelihoods = []
    with open(finename,'r') as file:
        normal_likelihoods = file.readlines()
        # print(type(normal_likelihood[0]),len(normal_likelihood[0]),type(normal_likelihood[0][0]))
        for each_line in normal_likelihoods:
            # print("---------------" + str(cnt+1) + "-----------------" )
            normal_likelihood_first = each_line
            normal_likelihood_first = normal_likelihood_first[1:-2]
            normal_likelihood_first =  normal_likelihood_first.replace(' ',',')
            normal_likelihood_first_new = normal_likelihood_first.split(',')
            # print('normal_likelihood_first_new1:',normal_likelihood_first_new)
            normal_likelihood_first_new = [item for item in normal_likelihood_first_new if item !=''] #取出多余空格
            # print('normal_likelihood_first_new2:',normal_likelihood_first_new)
            normal_likelihood_first_new = [float(item) for item in normal_likelihood_first_new]
            all_abnormal_likelihoods.append(normal_likelihood_first_new)
            # print("old:",normal_likelihood_first_new,type(normal_likelihood_first_new))
            # del normal_likelihood_first_new[cnt] # 删除训练中的数据位
            # print("old_deled:",normal_likelihood_first_new,type(normal_likelihood_first_new))
            # mean = round(np.mean(normal_likelihood_first_new),3)
            # std_dev = round(np.std(normal_likelihood_first_new),3)
            # print("mean:",mean,"std_dev:",std_dev)
            cnt += 1
    return all_abnormal_likelihoods

def cal_abnormal_sts2():
    all_abnormal_likelihoods = []
    finename = "likelihood_abnormal2c_1.txt"
    with open(finename,'r') as file:
        normal_likelihoods = file.readlines()
        for each_line in normal_likelihoods:
            normal_likelihood_first = each_line[2:-3]
            # print("old:",normal_likelihood_first,type(normal_likelihood_first))
            normal_likelihood_first =  normal_likelihood_first.replace(' ','')
            normal_likelihood_first_new = normal_likelihood_first.split(',')
            # print('normal_likelihood_first_new1:',normal_likelihood_first_new)
            normal_likelihood_first_new = [item for item in normal_likelihood_first_new if item !=''] #取出多余空格
            # print('normal_likelihood_first_new2:',normal_likelihood_first_new)
            # print("old:",normal_likelihood_first_new,type(normal_likelihood_first_new))
            normal_likelihood_first_new = [float(item) for item in normal_likelihood_first_new]
            all_abnormal_likelihoods.append(normal_likelihood_first_new)
    return all_abnormal_likelihoods

def cal_normal_sts():
    finename = "likelihood_normal2c_1.txt"
    cnt = 0
    all_likelihoods = []
    all_abnormal_likelihoods = cal_abnormal_sts2()
    # print("abnormal data:",all_abnormal_likelihoods)
    with open(finename,'r') as file:
        normal_likelihoods = file.readlines()
        # print(type(normal_likelihood[0]),len(normal_likelihood[0]),type(normal_likelihood[0][0]))
        for each_line in normal_likelihoods:
            this_mode_abnormal_likelihoods_deltas = []
            print("---------------" + str(cnt+1) + "-----------------" ) # 第几个模型
            normal_likelihood_first = each_line
            normal_likelihood_first = normal_likelihood_first[2:-3]
            # print("old:",normal_likelihood_first,type(normal_likelihood_first))
            normal_likelihood_first =  normal_likelihood_first.replace(' ','')
            normal_likelihood_first_new = normal_likelihood_first.split(',')
            # print('normal_likelihood_first_new1:',normal_likelihood_first_new)
            normal_likelihood_first_new = [item for item in normal_likelihood_first_new if item !=''] #取出多余空格
            # print('normal_likelihood_first_new2:',normal_likelihood_first_new)
            # print("old:",normal_likelihood_first_new,type(normal_likelihood_first_new))
            normal_likelihood_first_new = [float(item) for item in normal_likelihood_first_new]
            test_normal_E = normal_likelihood_first_new[cnt]
            # print("old:",normal_likelihood_first_new,type(normal_likelihood_first_new))
            del normal_likelihood_first_new[cnt] # 删除训练中的数据位
            # print("old_deled:",normal_likelihood_first_new,type(normal_likelihood_first_new))
            mean = round(np.mean(normal_likelihood_first_new),3)
            std_dev = round(np.std(normal_likelihood_first_new),3)
            this_mode_abnormal_likelihood = all_abnormal_likelihoods[cnt]
            test_normal_difference = abs(test_normal_E - mean) # 正常数据测试
            test_normal_deltata = round(test_normal_difference / std_dev,3)
            print("mean:",mean,"std_dev:",std_dev," test normal E:",test_normal_deltata)
            this_mode_normal_likelihoods_deltas = []
            for each_likelihood_E in normal_likelihood_first_new:
                difference = abs(each_likelihood_E - mean)
                each_delta = round(difference / std_dev,3)
                this_mode_normal_likelihoods_deltas.append(each_delta)
            print("this mode normal likelihoods:",this_mode_normal_likelihoods_deltas)
            for each_likelihood_E in this_mode_abnormal_likelihood:
                difference = abs(each_likelihood_E - mean)
                each_delta = round(difference / std_dev,3)
                this_mode_abnormal_likelihoods_deltas.append(each_delta)
            print("this mode abnormal likelihoods:",this_mode_abnormal_likelihoods_deltas)
            
            
            # noraml_likelihood_np = read_pickle('Loglikelihood_noraml')
            # abnoraml_likelihood_np = read_pickle('Loglikelihood_abnoraml')
            plt.figure()
            this_mode = '2c_kdeplot_2'
            fig_likelihood_path = 'fig_likelihoods' + '/' + this_mode
            if not os.path.exists(fig_likelihood_path):
                os.makedirs(fig_likelihood_path)
            noraml_likelihood_np = normal_likelihood_first_new
            abnoraml_likelihood_np = this_mode_abnormal_likelihood
            # noraml_likelihood_np = this_mode_normal_likelihoods_deltas
            # abnoraml_likelihood_np = this_mode_abnormal_likelihoods_deltas
            
            # fig data frequency
            frequency_fig_bool = False
            if frequency_fig_bool:
                cnt_normal = Counter(noraml_likelihood_np)
                cnt_abnormal = Counter(abnoraml_likelihood_np)
    
                percent_normal = {key:round(value/sum(cnt_normal.values()),3) for key, value in cnt_normal.items()}
                percent_abnormal = {key:round(value/sum(cnt_abnormal.values()),3) for key, value in cnt_abnormal.items()}
                # print(percent_normal)
                x_normal = list(percent_normal.keys())
                y_normal = list(percent_normal.values())
                x_abnormal = list(percent_abnormal.keys())
                y_abnormal = list(percent_abnormal.values())
                # print(percent_abnormal)
                
                plt.plot(x_normal, y_normal, 'b*-', alpha=0.5, linewidth=1, label='normal')#'bo-'表示蓝色实线，数据点实心原点标注
                plt.plot(x_abnormal, y_abnormal, 'rs--', alpha=0.5, linewidth=1, label='abnormal')#'bo-'表示蓝色实线，数据点实心原点标注
                ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，

                plt.legend()  #显示上面的label
                plt.xlabel('Delta') #x_label
                plt.ylabel('Frequency')#y_label
            
                #plt.ylim(-1,1)#仅设置y轴坐标范围
                plt.savefig(fig_likelihood_path + '/' + str(cnt+1) + '.png')
                
            hist_fig_bool = True
            if hist_fig_bool:
                # cnt_normal = Counter(noraml_likelihood_np)
                # cnt_abnormal = Counter(abnoraml_likelihood_np)
    
                # percent_normal = {key:round(value/sum(cnt_normal.values()),3) for key, value in cnt_normal.items()}
                # percent_abnormal = {key:round(value/sum(cnt_abnormal.values()),3) for key, value in cnt_abnormal.items()}
                # print(percent_normal)
                x_normal = noraml_likelihood_np
                x_abnormal = abnoraml_likelihood_np
                # y_normal = list(percent_normal.values())
                # x_abnormal = list(percent_abnormal.keys())
                # y_abnormal = list(percent_abnormal.values())
                # # print(percent_abnormal)
                
                # plt.plot(x_normal, y_normal, 'b*-', alpha=0.5, linewidth=1, label='normal')#'bo-'表示蓝色实线，数据点实心原点标注
                # plt.plot(x_abnormal, y_abnormal, 'rs--', alpha=0.5, linewidth=1, label='abnormal')#'bo-'表示蓝色实线，数据点实心原点标注
                ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
                # plt.hist(x_normal, bins= 'auto', facecolor='black',edgecolor='red', alpha=0.5)
                # plt.hist(x_abnormal, bins='auto', facecolor='green',edgecolor='yellow', alpha=0.5)
                sns.kdeplot(x_normal,fill=True, color='blue', alpha=0.7,label='normal')
                sns.kdeplot(x_abnormal,fill=True, color='red', alpha=0.7, label='abnormal')
                plt.legend()  #显示上面的label
                plt.xlabel('Data') #x_label
                plt.ylabel('Frequency')#y_label
            
                #plt.ylim(-1,1)#仅设置y轴坐标范围
                plt.savefig(fig_likelihood_path + '/' + str(cnt+1) + '.png')
                # plt.show()
            
            cnt += 1


if __name__ == "__main__":
    
    
    cal_normal_sts()
    # cal_abnormal_sts2
    
    
    
    # noraml_likelihood_np = read_pickle('Loglikelihood_noraml')
    # abnoraml_likelihood_np = read_pickle('Loglikelihood_abnoraml')
    # cnt_normal = Counter(noraml_likelihood_np)
    # cnt_abnormal = Counter(abnoraml_likelihood_np)
    
    # percent_normal = {key:round(value/sum(cnt_normal.values()),3) for key, value in cnt_normal.items()}
    # percent_abnormal = {key:round(value/sum(cnt_abnormal.values()),3) for key, value in cnt_abnormal.items()}
    # print(percent_normal)
    # x_normal = list(percent_normal.keys())
    # y_normal = list(percent_normal.values())
    # x_abnormal = list(percent_abnormal.keys())
    # y_abnormal = list(percent_abnormal.values())
    # print(percent_abnormal)
    
    # plt.plot(x_normal, y_normal, 'b*--', alpha=0.5, linewidth=1, label='normal')#'bo-'表示蓝色实线，数据点实心原点标注
    # plt.plot(x_abnormal, y_abnormal, 'rs--', alpha=0.5, linewidth=1, label='abnormal')#'bo-'表示蓝色实线，数据点实心原点标注
    # ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，

    # plt.legend()  #显示上面的label
    # plt.xlabel('Likelihood') #x_label
    # plt.ylabel('Frequency')#y_label
 
    # #plt.ylim(-1,1)#仅设置y轴坐标范围
    # plt.show()


