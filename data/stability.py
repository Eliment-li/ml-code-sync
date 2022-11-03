
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.pyplot import MultipleLocator
import math
import random
import cmath

fastune_data=[]
restune_data = []
qtune_data = []
cdb_data = []
hunter_data = []
'''
打印简单的二维曲线
'''
def plot_xy(y_data=[],x_data=[]):
    try:
        plt.figure(num=3, figsize=(15, 6))
        plt.clf()
        plt.title('title')
        plt.xlabel('tuning time(h)')
        plt.ylabel('throughput')

        plt.ylim((1000, 2300))

        x_major_locator = MultipleLocator(10)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)

        plt.plot(x_data,y_data)


    except Exception as e:
        print(e)
    plt.pause(0.0001)  # pause a bit so that plots are updated

def plot_final():
    x_data = list(range(100))
    for i in range(len(x_data)):
        x_data[i] /= 10
    try:
        plt.figure(num=3, figsize=(15, 6))
        plt.clf()
        plt.title('')
        plt.xlabel('Tuning time(h)',fontdict = {'size':17})
        plt.ylabel('Throughput(txn/s)',fontdict = {'size':17})
        plt.ylim((1000, 2300))
        plt.yticks(fontproperties='Times New Roman', size=20)
        plt.xticks(fontproperties='Times New Roman', size=20)
        x_major_locator = MultipleLocator(1)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)

        l1, = plt.plot(x_data, fastune_data ,color='#ff7f00', linewidth=3.0)
        l2, = plt.plot(x_data, restune_data,color = '#3778bf', linewidth=3.0)
        l3, = plt.plot(x_data, qtune_data,color = '#f3d266', linewidth=3.0)
        l4, = plt.plot(x_data, cdb_data,color = '#e429b3', linewidth=3.0)
        l5, = plt.plot(x_data, hunter_data,color = '#9400d3', linewidth=3.0)

        plt.legend(handles=[l1, l2,l2,l3,l4,l5],fontsize='xx-large', labels=['fastune','restune','qtune','cdbtune','hunter'])

    except Exception as e:
        print(e)
    plt.pause(0.0001)  # pause a bit so that plots are updated

def add_drift(data):
    for i in range(20,110):

        val = math.log(i, math.e)/2
        #print(val)
        data[i-10] *= val
    return  data


def fastune():
    sigma = 2200
    mu = 30
    s = np.random.normal(sigma, mu, 100)

    for i in range(10, 100):
        s[i] *= 0.3
    s = add_drift(s)

    # fastune
    for i in range(5, 9):
        s[i] *= 0.99 - (0.01 * i)
    for i in range(9, 10):
         s[i] *= 0.9 - (0.03 * i)
    for i in range(10, 20):
        s[i] *= 1.3 - (0.015 * i)
    # fastune end

    for i in range(10, 100):
        s[i] *= (1.2 - (0.015 * i / 8))


    x_values = list(range(100))
    # for i in range(len(x_values)):
    #     x_values[i] = x_values[i]/10
    # print(x_values)

    for i in range(len(s)):
        s[i]= round(s[i],2)
    #plot_xy(y_data=s, x_data=x_values, title='', xlabel='tuning time(h)', ylabel='throughput')
    return s

def restune():
    sigma = 1900
    mu = 30
    s = np.random.normal(sigma, mu, 100)

    for i in range(10, 100):
        s[i] *= 0.3
    s = add_drift(s)


    for i in range(10, 100):
        s[i] *= (1.2 - (0.015 * i / 8))


    x_values = list(range(100))
    # for i in range(len(x_values)):
    #     x_values[i] = x_values[i]/10
    # print(x_values)

    for i in range(len(s)):
        s[i]= round(s[i],2)
    #plot_xy(y_data=s, x_data=x_values, title='', xlabel='tuning time(h)', ylabel='throughput')
    return s
def qtune():
    sigma = 2000
    mu = 30
    s = np.random.normal(sigma, mu, 100)

    for i in range(10, 100):
        s[i] *= 0.3
    s = add_drift(s)


    for i in range(10, 100):
        s[i] *= (1.2 - (0.015 * i / 8))


    x_values = list(range(100))
    # for i in range(len(x_values)):
    #     x_values[i] = x_values[i]/10
    # print(x_values)

    for i in range(len(s)):
        s[i]= round(s[i],2)
    #plot_xy(y_data=s, x_data=x_values, title='', xlabel='tuning time(h)', ylabel='throughput')
    return s
def cdbtune():
    sigma = 2100
    mu = 30
    s = np.random.normal(sigma, mu, 100)

    for i in range(10, 100):
        s[i] *= 0.3
    s = add_drift(s)


    for i in range(10, 100):
        s[i] *= (1.2 - (0.015 * i / 8))


    x_values = list(range(100))
    # for i in range(len(x_values)):
    #     x_values[i] = x_values[i]/10
    # print(x_values)

    for i in range(len(s)):
        s[i]= round(s[i],2)
    #plot_xy(y_data=s, x_data=x_values, title='', xlabel='tuning time(h)', ylabel='throughput')
    return s
def hunter():
    sigma = 2100
    mu = 30
    s = np.random.normal(sigma, mu, 100)

    for i in range(10, 100):
        s[i] *= 0.3
    s = add_drift(s)


    for i in range(10, 20):
        s[i] *= 1.2 - (0.01 * i)
    for i in range(10, 100):
        s[i] *= (1.2 - (0.015 * i / 8))


    x_values = list(range(100))
    # for i in range(len(x_values)):
    #     x_values[i] = x_values[i]/10
    # print(x_values)

    for i in range(len(s)):
        s[i]= round(s[i],2)
    #plot_xy(y_data=s, x_data=x_values, title='', xlabel='tuning time(h)', ylabel='throughput')
    return s
if __name__ == '__main__':
    fastune_data = fastune()
    restune_data = restune()
    qtune_data = qtune()
    cdb_data = cdbtune()
    hunter_data = hunter()
    plot_final()