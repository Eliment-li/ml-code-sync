import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

if __name__ == '__main__':

    #data = pd.read_csv("E://ML2022//HW1//feature_train.csv")
    data = pd.read_csv("E://ML2022//HW1//covid.train.csv")
    X = data.iloc[:,0:93]  #independent columns
    y = data.iloc[:,-1]    #target column i.e price range
    #apply SelectKBest class to extract top 10 best features
    bestfeatures = SelectKBest(score_func=f_regression, k=93)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization

    pd.set_option('display.max_rows', None)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)

    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    print(featureScores.nlargest(93,'Score'))  #print 10 best features