import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

if __name__ == '__main__':
    data = pd.read_csv("E://ML2022//HW1//feature_train.csv")
    X = data.iloc[:,0:20]  #independent columns
    y = data.iloc[:,-1]    #target column i.e price range
    #get correlations of each features in dataset
    corrmat = data.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20,20))
    #plot heat map
    g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
    plt.show()