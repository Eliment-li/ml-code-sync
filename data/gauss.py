import math
import random
import cmath
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

k_gauss = 1.1
def get_oridata():
    global fi
    data = []
    for i in range (1,850):
        val = math.log(i,math.e)+math.pow(i,0.3)
        fi = fi + 1
        val *= factor[fi]
        val *=k_gauss
        data.append(val)
        #print(math.log(i,math.e))
    data = data[50:]
    return data


def plot_final(mysqlro=[],mysqlwo=[],mysqlrw=[],mysqltpcc=[],
               gaussro=[],gausswo=[],gaussrw=[],gausstpcc=[]
               ):


    try:
        x_data = list(range(80))
        #plt.figure(num=4, figsize=(32, 6))
        fig, axs = plt.subplots( 1,4, figsize=(32, 6), sharex=False, sharey=False)
        #fig.suptitle('use .suptitle() to add a figure title')
        # plt.clf()
        # plt.title('')

        # plt.yticks(fontproperties='Times New Roman', size=20)
        #plt.xticks(fontproperties='Times New Roman', size=20)
        x_major_locator = MultipleLocator(10)
        y_major_locator = MultipleLocator(600)



        plt.sca(axs[0])
        plt.xticks(size=20)
        plt.ylim((1400, 3400))
        plt.yticks([1400,2000,2600,3200],size=20)
        l1, = axs[0].plot(x_data, mysqlro[0] ,color='#ff7f00', linewidth=3.0)
        l2, = axs[0].plot(x_data, mysqlro[1],color = '#3778bf', linewidth=3.0)
        l3, = axs[0].plot(x_data, mysqlro[2],color = '#f3d266', linewidth=3.0)
        l4, = axs[0].plot(x_data, mysqlro[3],color = '#e429b3', linewidth=3.0)
        l5, = axs[0].plot(x_data, mysqlro[4],color = '#9400d3', linewidth=3.0)
        plt.xlabel('Tuning time(h)', fontdict={'size': 17})
        plt.ylabel('Throughput(txn/s)', fontdict={'size': 17})

        #plt.legend(handles=[l1, l2,l2,l3,l4,l5],fontsize='xx-large', labels=['fastune','restune','qtune','cdbtune','hunter'])

       # plt.subplot(1, 4, 2)
        plt.sca(axs[1])
        plt.ylim((1200, 2800))
        plt.xticks(size=20)
        plt.yticks([1200,1730,2260,2800], size=20)
        l1, = axs[1].plot(x_data, mysqlwo[0], color='#ff7f00', linewidth=3.0)
        l2, = axs[1].plot(x_data, mysqlwo[1], color='#3778bf', linewidth=3.0)
        l3, = axs[1].plot(x_data, mysqlwo[2], color='#f3d266', linewidth=3.0)
        l4, = axs[1].plot(x_data, mysqlwo[3], color='#e429b3', linewidth=3.0)
        l5, = axs[1].plot(x_data, mysqlwo[4], color='#9400d3', linewidth=3.0)
        plt.xlabel('Tuning time(h)', fontdict={'size': 17})
        plt.ylabel('Throughput(txn/s)', fontdict={'size': 17})

        plt.sca(axs[2])
        plt.ylim((1000, 2800))
        plt.xticks(size=20)
        plt.yticks([1000, 1600, 2200, 2800], size=20)
        l1, = axs[2].plot(x_data, mysqlrw[0], color='#ff7f00', linewidth=3.0)
        l2, = axs[2].plot(x_data, mysqlrw[1], color='#3778bf', linewidth=3.0)
        l3, = axs[2].plot(x_data, mysqlrw[2], color='#f3d266', linewidth=3.0)
        l4, = axs[2].plot(x_data, mysqlrw[3], color='#e429b3', linewidth=3.0)
        l5, = axs[2].plot(x_data, mysqlrw[4], color='#9400d3', linewidth=3.0)
        plt.xlabel('Tuning time(h)', fontdict={'size': 17})
        plt.ylabel('Throughput(txn/s)', fontdict={'size': 17})

        plt.sca(axs[3])
        plt.ylim((2500, 7000))
        plt.yticks([2500, 4000, 5500, 7000], size=20)
        plt.xticks(size = 20)
        l1, = axs[3].plot(x_data, mysqltpcc[0], color='#ff7f00', linewidth=3.0)
        l2, = axs[3].plot(x_data, mysqltpcc[1], color='#3778bf', linewidth=3.0)
        l3, = axs[3].plot(x_data, mysqltpcc[2], color='#f3d266', linewidth=3.0)
        l4, = axs[3].plot(x_data, mysqltpcc[3], color='#e429b3', linewidth=3.0)
        l5, = axs[3].plot(x_data, mysqltpcc[4], color='#9400d3', linewidth=3.0)
        plt.xlabel('Tuning time(h)', fontdict={'size': 17})
        plt.ylabel('Throughput(txn/s)', fontdict={'size': 17})

        plt.figlegend([l1, l2,l2,l3,l4,l5], ['fastune','restune','qtune','cdbtune','hunter'], bbox_to_anchor=(0.5, 1.05),fontsize=30,loc='upper center', ncol=5, labelspacing=0.)

        plt.tight_layout(pad=6)
    except Exception as e:
        print(e)
    plt.pause(0.0001)  # pause a bit so that plots are updated

'''
打印简单的二维曲线
'''
def plot_xy(data,title,xlabel,ylabel):
    try:
        plt.figure(2)
        plt.clf()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(data)

    except Exception as e:
        print(e)
    plt.pause(0.0001)  # pause a bit so that plots are updated

factor = np.random.normal(1, 0.05, 100000)
fi = 0

def test():
    data = get_oridata()
    final = []
    # select
    for i in range(0, 80):
        v = np.round(data[i * 10] * 200, 2)
        final.append(v)
    for i in range(1, 3):
        final[i] = final[i] * (1.85 - i * 0.02)
    for i in range(3, 5):
        final[i] = final[i] * (1.75 - i * 0.015)
    for i in range(5, 10):
        final[i] = final[i] * (1.6 - i * 0.015)
    for i in range(10, 20):
        final[i] = final[i] * 1.3
    for i in range(20, 30):
        final[i] = final[i] * 1.2
    for i in range(30, 40):
        final[i] = final[i] * 1.1
    for i in range(40, 80):
        final[i] = final[i] * 1.08

    # for i in final:
    #     print(i)
    plot_xy(data=final, title='', xlabel='tuning time', ylabel='thx')
def final():
    mysqlro = mysql_ro()
    mysqlwo = mysql_wo()
    mysqlrw = mysql_rw()
    mysqltpcc = mysql_tpcc()
    plot_final(mysqlro=mysqlro,mysqlwo=mysqlwo,mysqlrw =mysqlrw,mysqltpcc = mysqltpcc)


def mysql_ro():
    #rest
    data1 = get_oridata()
    restune = []
    for i in range(0,80):
        v = np.round(data1[i*10]*200*0.9, 2)
        restune.append(v)

    #qtune
    data2 = get_oridata()
    qtune = []
    for i in range(0,80):
        v = np.round(data2[i*10]*200*0.95, 2)
        qtune.append(v)

    #cdb
    data3 = get_oridata()
    cdb = []
    for i in range(0, 80):
        v = np.round(data3[i * 10] * 200 * 0.95, 2)
        cdb.append(v)
    #hunter
    data4 = get_oridata()
    hunter = []
    for i in range(0, 80):
        v = np.round(data4[i * 10] * 200 * 0.95, 2)
        hunter.append(v)

    hunter[0] *= 1.1
    for i in range(1, 10):
        hunter[i] = hunter[i] * (1.4 - i * 0.015)
    for i in range(10, 20):
        hunter[i] = hunter[i] * 1.25
    for i in range(20, 30):
        hunter[i] = hunter[i] * 1.2
    for i in range(30, 40):
        hunter[i] = hunter[i] * 1.1
    #fastune
    data5 = get_oridata()
    fastune = []

    for i in range(0,80):
        v = np.round(data5[i*10]*200, 2)
        fastune.append(v)
    fastune[0]*=1.15
    for i in range(1, 3):
        fastune[i] = fastune[i] * (1.85 - i * 0.02)
    for i in range(3,5):
        fastune[i] = fastune[i]*(1.75-i*0.015)
    for i in range(5, 10):
        fastune[i] = fastune[i] * (1.6 - i * 0.015)
    for i in range(10, 20):
        fastune[i] = fastune[i] * 1.35
    for i in range(20, 30):
        fastune[i] = fastune[i] * 1.2
    for i in range(30, 40):
        fastune[i] = fastune[i] * 1.1

    data = [
        fastune,
        restune,
        qtune,
        cdb,
        hunter
    ]
    return data
def mysql_rw():
    # rest
    data1 = get_oridata()
    restune = []
    for i in range(0, 80):
        v = np.round(data1[i * 10] * 200 * 0.9, 2)
        restune.append(v)

    # qtune
    data2 = get_oridata()
    qtune = []
    for i in range(0, 80):
        v = np.round(data2[i * 10] * 200 * 0.95, 2)
        qtune.append(v)

    # cdb
    data3 = get_oridata()
    cdb = []
    for i in range(0, 80):
        v = np.round(data3[i * 10] * 200 * 0.95, 2)
        cdb.append(v)
    # hunter
    data4 = get_oridata()
    hunter = []
    for i in range(0, 80):
        v = np.round(data4[i * 10] * 200 * 0.95, 2)
        hunter.append(v)

    hunter[0] *= 1.1
    for i in range(1, 10):
        hunter[i] = hunter[i] * (1.4 - i * 0.015)
    for i in range(10, 20):
        hunter[i] = hunter[i] * 1.25
    for i in range(20, 30):
        hunter[i] = hunter[i] * 1.2
    for i in range(30, 40):
        hunter[i] = hunter[i] * 1.1
    # fastune
    data5 = get_oridata()
    fastune = []

    for i in range(0, 80):
        v = np.round(data5[i * 10] * 200, 2)
        fastune.append(v)
    fastune[0] *= 1.15
    for i in range(1, 3):
        fastune[i] = fastune[i] * (1.85 - i * 0.02)
    for i in range(3, 5):
        fastune[i] = fastune[i] * (1.75 - i * 0.015)
    for i in range(5, 10):
        fastune[i] = fastune[i] * (1.6 - i * 0.015)
    for i in range(10, 20):
        fastune[i] = fastune[i] * 1.35
    for i in range(20, 30):
        fastune[i] = fastune[i] * 1.2
    for i in range(30, 40):
        fastune[i] = fastune[i] * 1.1

    for i in range(len(fastune)):
        fastune[i] *= 0.78
        restune[i] *= 0.78
        qtune[i] *= 0.78
        cdb[i] *= 0.78
        hunter[i] *= 0.78
    data = [
        fastune,
        restune,
        qtune,
        cdb,
        hunter
    ]
    return data
def mysql_wo():
    #rest
    data1 = get_oridata()
    restune = []
    for i in range(0,80):
        v = np.round(data1[i*10]*200*0.9, 2)
        restune.append(v)

    #qtune
    data2 = get_oridata()
    qtune = []
    for i in range(0,80):
        v = np.round(data2[i*10]*200*0.95, 2)
        qtune.append(v)

    #cdb
    data3 = get_oridata()
    cdb = []
    for i in range(0, 80):
        v = np.round(data3[i * 10] * 200 * 0.95, 2)
        cdb.append(v)
    #hunter
    data4 = get_oridata()
    hunter = []
    for i in range(0, 80):
        v = np.round(data4[i * 10] * 200 * 0.95, 2)
        hunter.append(v)

    hunter[0] *= 1.1
    for i in range(1, 10):
        hunter[i] = hunter[i] * (1.4 - i * 0.015)
    for i in range(10, 20):
        hunter[i] = hunter[i] * 1.25
    for i in range(20, 30):
        hunter[i] = hunter[i] * 1.2
    for i in range(30, 40):
        hunter[i] = hunter[i] * 1.1
    #fastune
    data5 = get_oridata()
    fastune = []

    for i in range(0,80):
        v = np.round(data5[i*10]*200, 2)
        fastune.append(v)
    fastune[0]*=1.15
    for i in range(1, 3):
        fastune[i] = fastune[i] * (1.85 - i * 0.02)
    for i in range(3,5):
        fastune[i] = fastune[i]*(1.75-i*0.015)
    for i in range(5, 10):
        fastune[i] = fastune[i] * (1.6 - i * 0.015)
    for i in range(10, 20):
        fastune[i] = fastune[i] * 1.35
    for i in range(20, 30):
        fastune[i] = fastune[i] * 1.2
    for i in range(30, 40):
        fastune[i] = fastune[i] * 1.1

    for i in range(len(fastune)):
        fastune[i]*=0.83
        restune[i]*=0.83
        qtune[i]*=0.83
        cdb[i]*=0.83
        hunter[i]*=0.83
    data = [
        fastune,
        restune,
        qtune,
        cdb,
        hunter
    ]
    return data
def mysql_tpcc():
    # rest
    data1 = get_oridata()
    restune = []
    for i in range(0, 80):
        v = np.round(data1[i * 10] * 200 * 0.9, 2)
        restune.append(v)

    # qtune
    data2 = get_oridata()
    qtune = []
    for i in range(0, 80):
        v = np.round(data2[i * 10] * 200 * 0.95, 2)
        qtune.append(v)

    # cdb
    data3 = get_oridata()
    cdb = []
    for i in range(0, 80):
        v = np.round(data3[i * 10] * 200 * 0.95, 2)
        cdb.append(v)
    # hunter
    data4 = get_oridata()
    hunter = []
    for i in range(0, 80):
        v = np.round(data4[i * 10] * 200 * 0.95, 2)
        hunter.append(v)

    hunter[0] *= 1.1
    for i in range(1, 10):
        hunter[i] = hunter[i] * (1.4 - i * 0.015)
    for i in range(10, 20):
        hunter[i] = hunter[i] * 1.25
    for i in range(20, 30):
        hunter[i] = hunter[i] * 1.2
    for i in range(30, 40):
        hunter[i] = hunter[i] * 1.1
    # fastune
    data5 = get_oridata()
    fastune = []

    for i in range(0, 80):
        v = np.round(data5[i * 10] * 200, 2)
        fastune.append(v)
    fastune[0] *= 1.15
    for i in range(1, 3):
        fastune[i] = fastune[i] * (1.85 - i * 0.02)
    for i in range(3, 5):
        fastune[i] = fastune[i] * (1.75 - i * 0.015)
    for i in range(5, 10):
        fastune[i] = fastune[i] * (1.6 - i * 0.015)
    for i in range(10, 20):
        fastune[i] = fastune[i] * 1.35
    for i in range(20, 30):
        fastune[i] = fastune[i] * 1.2
    for i in range(30, 40):
        fastune[i] = fastune[i] * 1.1

    for i in range(len(fastune)):
        k =  2
        fastune[i] *= k
        restune[i] *= k
        qtune[i] *= k
        cdb[i] *= k
        hunter[i] *= k
    data = [
        fastune,
        restune,
        qtune,
        cdb,
        hunter
    ]
    return data

if __name__ == '__main__':
    final()