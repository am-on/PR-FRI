import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error

def learnPredictScore(model, trainX, trainY, testX, testY):
    model.fit(trainX, trainY)

    # predict ratings for current user
    prediction = model.predict(testX)

    # MAE
    return mean_absolute_error(prediction, testY)


dfRatings = pd.read_csv('../data/ratings.csv')
dfMovies = pd.read_csv('../data/movies.csv')

# transform df
df = dfRatings.pivot(index='movieId', columns='userId', values='rating')

# get users with more than 100 ratings
dfU = df.dropna(axis=1, how='any', thresh=100, subset=None)

# drop movies with less than 100 ratings
df.dropna(axis=0, how='any', thresh=100, subset=None, inplace=True)

# drop users with less than 100 ratings
df.drop([col for col in df.columns if col not in dfU.columns], axis=1, inplace=True)


# replace null values with 0
df = df.fillna(0)

linearScores = []
lassoScores = []
ridgeScores = []

for user in df.columns:
    for x in range(3):
        uDf = df[df.loc[:, user] > 0]

        # split df into training and test sets
        trainDf = uDf.sample(frac=.75)
        testDf = uDf[~uDf.index.isin(trainDf.index)]

        # get ratings of current user
        trainUserDf = trainDf.loc[:, user].copy()
        testUserDf = testDf.loc[:, user].copy()

        # drop current user from training and test sets
        trainDf = trainDf.drop(user, 1)
        testDf = testDf.drop(user, 1)

        # create models
        linear = linear_model.LinearRegression()
        lasso = linear_model.Lasso(alpha=0.2)
        ridge = linear_model.BayesianRidge()

        # train models and get model score for current user
        linearScores.append(learnPredictScore(linear, trainDf, trainUserDf, testDf, testUserDf))
        lassoScores.append(learnPredictScore(lasso, trainDf, trainUserDf, testDf, testUserDf))
        ridgeScores.append(learnPredictScore(ridge, trainDf, trainUserDf, testDf, testUserDf))



print("Linear regression: {0},\nLasso: {1},\nBayesianRidge: {2}".format(np.mean(linearScores), np.mean(lassoScores), np.mean(ridgeScores)))
