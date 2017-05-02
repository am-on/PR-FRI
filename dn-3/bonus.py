import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error

def learnPredict(model, trainX, trainY, testX, testY):
    model.fit(trainX, trainY)

    # predict ratings for current user
    prediction = model.predict(testX)

    return prediction


dfRatings = pd.read_csv('../data/ratings.csv')
dfMovies = pd.read_csv('../data/movies.csv')

# transform df
df = dfRatings.pivot(index='movieId', columns='userId', values='rating')


# drop movies with less than 100 ratings
df.dropna(axis=0, how='any', thresh=100, subset=None, inplace=True)

# drop users with less than 100 ratings
df.dropna(axis=1, how='any', thresh=100, subset=None, inplace=True)


# replace null values with 0
df = df.fillna(0)

myRatings = [4, 5, 0, 0, 0, 0, 0, 0 ,0, 4, 0, 0, 0, 0,
             3, 0, 0, 0, 0, 1, 0, 5, 0, 0, 0, 0, 0, 0,
             4, 0, 5, 0, 0, 4, 4, 2, 0, 4, 5, 5, 5, 0, 5]
myRLen = len(myRatings)

for x in range(len(df)-myRLen):
    myRatings.append(0)

user = 9999

df2 = pd.DataFrame({user:myRatings}, index = df.index)

df = pd.concat([df,df2], axis=1)



# split df into training and test sets
trainDf = df[:myRLen]
testDf = df[~df.index.isin(trainDf.index)]

# get ratings of current user
trainUserDf = trainDf.loc[:, user].copy()
testUserDf = testDf.loc[:, user].copy()

# drop current user from training and test sets
trainDf = trainDf.drop(user, 1)
testDf = testDf.drop(user, 1)

# create models
model = linear_model.Lasso()

# train models and get model score for current user

pred = learnPredict(model, trainDf, trainUserDf, testDf, testUserDf)

for x, rate in zip(testDf.index, pred):
    print(dfMovies.loc[dfMovies['movieId'] == x].title, rate)

print(len(df))










