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

if __name__ == "__main__":
    
    noraml_likelihood_np = read_pickle('Loglikelihood_noraml')
    abnoraml_likelihood_np = read_pickle('Loglikelihood_abnoraml')
    cnt_normal = Counter(noraml_likelihood_np)
    cnt_abnormal = Counter(abnoraml_likelihood_np)
    
    percent_normal = {key:round(value/sum(cnt_normal.values()),3) for key, value in cnt_normal.items()}
    percent_abnormal = {key:round(value/sum(cnt_abnormal.values()),3) for key, value in cnt_abnormal.items()}
    print(percent_normal)
    x_normal = list(percent_normal.keys())
    y_normal = list(percent_normal.values())
    x_abnormal = list(percent_abnormal.keys())
    y_abnormal = list(percent_abnormal.values())
    print(percent_abnormal)
    
    plt.plot(x_normal, y_normal, 'b*--', alpha=0.5, linewidth=1, label='normal')#'bo-'表示蓝色实线，数据点实心原点标注
    plt.plot(x_abnormal, y_abnormal, 'rs--', alpha=0.5, linewidth=1, label='abnormal')#'bo-'表示蓝色实线，数据点实心原点标注
    ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，

    plt.legend()  #显示上面的label
    plt.xlabel('Likelihood') #x_label
    plt.ylabel('Frequency')#y_label
 
    #plt.ylim(-1,1)#仅设置y轴坐标范围
    plt.show()


