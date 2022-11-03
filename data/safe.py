import matplotlib.pyplot as plt
import numpy as np
#rest  qtune cdb hunter fastune
tpcc1 = [101, 349, 306, 95, 6,
         8, 21, 13, 3, 6]


Wikipedia1=[99, 266, 319, 59, 6,
            9, 22, 18, 19,6]

SmallBank1=[59, 311, 299, 88, 6,
            8, 24, 26, 13, 6]


color = ['#0784d5', '#fb711e', '#c1c020', '#00996b', '#6a0d9a']

def bank():

    plt.figure(figsize=(11,9))
    plt.bar(range(len(SmallBank1)), SmallBank1,color=color)
    labels = ['#Dangerous',  '#Failure']
    x = [2, 7]
    y = [0, 100, 200, 300,400]

    xdata = [0,1,2,3,4,5,6,7,8,9]
    ydata = SmallBank1
    ydata[4]=2
    ydata[9]=0
    # plt.tick_params(
    #     axis='x',  # changes apply to the x-axis
    #     which='both',  # both major and minor ticks are affected
    #     bottom=False,  # ticks along the bottom edge are off
    #     top=False,  # ticks along the top edge are off
    #     labelbottom=False)  # labels along the bottom edge are off
    plt.xticks(x, labels,fontsize = 30)
    plt.yticks(y, fontsize = 30)
    plt.ylabel('Count', fontdict={'size': 30})
    for a, b in zip(xdata, SmallBank1):
        plt.text(a, b+2,
                 b,
                 ha='center',
                 va='bottom',
                 fontsize=30
                 )
    colors = {'Restune':'#0784d5', 'QTune':'#fb711e', 'CDBTune':'#c1c020','Hunter':'#00996b','Fastune':'#6a0d9a'}
    labels = list(colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    plt.legend(handles, labels,fontsize=30)
    plt.show()
def wiki():
    plt.figure(figsize=(11,9))
    plt.bar(range(len(Wikipedia1)), Wikipedia1,color=color)
    labels = ['#Dangerous',  '#Failure']
    x = [2, 7]
    y = [0, 100, 200, 300,400]

    xdata = [0,1,2,3,4,5,6,7,8,9]
    ydata = Wikipedia1
    ydata[4] = 2
    ydata[9] = 0
    # plt.tick_params(
    #     axis='x',  # changes apply to the x-axis
    #     which='both',  # both major and minor ticks are affected
    #     bottom=False,  # ticks along the bottom edge are off
    #     top=False,  # ticks along the top edge are off
    #     labelbottom=False)  # labels along the bottom edge are off
    plt.xticks(x, labels,fontsize = 30)
    plt.yticks(y, fontsize = 30)
    plt.ylabel('Count', fontdict={'size': 30})
    for a, b in zip(xdata, Wikipedia1):
        plt.text(a, b+2,
                 b,
                 ha='center',
                 va='bottom',
                 fontsize=30
                 )
    colors = {'Restune':'#0784d5', 'QTune':'#fb711e', 'CDBTune':'#c1c020','Hunter':'#00996b','Fastune':'#6a0d9a'}
    labels = list(colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    plt.legend(handles, labels,fontsize=30)
    plt.show()

def tpcc():

    plt.figure(figsize=(11,9))
    plt.bar(range(len(tpcc1)), tpcc1,color=color)
    labels = ['#Dangerous',  '#Failure']
    x = [2, 7]
    y = [0, 100, 200, 300,400]

    xdata = [0,1,2,3,4,5,6,7,8,9]
    ydata = tpcc1
    ydata[4]=3
    ydata[5]=1
    ydata[8]=2
    ydata[9]=0
    # plt.tick_params(
    #     axis='x',  # changes apply to the x-axis
    #     which='both',  # both major and minor ticks are affected
    #     bottom=False,  # ticks along the bottom edge are off
    #     top=False,  # ticks along the top edge are off
    #     labelbottom=False)  # labels along the bottom edge are off
    plt.xticks(x, labels,fontsize = 30)
    plt.yticks(y, fontsize = 30)
    plt.ylabel('Count', fontdict={'size': 30})
    for a, b in zip(xdata, tpcc1):
        plt.text(a, b+2,
                 b,
                 ha='center',
                 va='bottom',
                 fontsize=30
                 )
    colors = {'Restune':'#0784d5', 'QTune':'#fb711e', 'CDBTune':'#c1c020','Hunter':'#00996b','Fastune':'#6a0d9a'}
    labels = list(colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    plt.legend(handles, labels,fontsize=30)
    plt.show()

if __name__ == '__main__':
    tpcc()
    wiki()
    bank()