import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

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

scoresCA = []
scoresF1 = []

for user in df.columns:
    for x in range(3):
        uDf = df[df.loc[:, user] > 0]

        # split df into training and test sets
        trainDf = uDf.sample(frac=.75)
        testDf = uDf[~uDf.index.isin(trainDf.index)]

        # get ratings of current user
        trainUserDf = trainDf.loc[:, user].copy()
        testUserDf = testDf.loc[:, user].copy()

        # binarize current user data
        trainUserDf = trainUserDf.apply(lambda x: 1 if x > 3 else 0)
        testUserDf = testUserDf.apply(lambda x: 1 if x > 3 else 0)

        # drop current user from training and test sets
        trainDf = trainDf.drop(user, 1)
        testDf = testDf.drop(user, 1)

        # create model
        model = svm.SVC()

        # skip building of model if only one class is present
        if len(set(trainUserDf)) < 2:
            continue

        # fit data to model
        model.fit(trainDf, trainUserDf)

        # predict
        prediction = model.predict(testDf)

        #prediction = [1 for x in prediction]

        # evaluate model
        scoresCA.append(accuracy_score(testUserDf, prediction))
        scoresF1.append(f1_score(testUserDf, prediction))


print("Klasifikacijska tocnost: {0}".format(np.mean(scoresCA)))
print("F1: {0}".format(np.mean(scoresF1)))
