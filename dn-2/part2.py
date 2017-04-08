def distance(x, y):
    """
    x, y: vectors (numpy arrays)
    """

    return np.sqrt(np.sum(x - y) ** 2)

def distance0(x, y):
    """
    x, y: vectors (numpy arrays)
    """
    d = 0
    for a, b in zip(x,y):
        if a != 0:
            d += 1

    return d/len(x)

class KMeans:
    def __init__(self, k=10, max_iter=100):
        """
        Initialize KMeans clustering model.

        :param k
            Number of clusters.
        :param max_iter
            Maximum number of iterations.
        """
        self.k = k
        self.max_iter = max_iter

    def fit(self, X):
        """
        Fit the Kmeans model to data.

        :param X
            Numpy array of shape (n, p)
            n: number of data examples
            p: number of features (attributes)

        :return
            labels: array of shape (n, ), cluster labels (0..k-1)
            centers: array of shape (k, p, )
        """

        n, p = X.shape
        labels = np.random.choice(range(self.k), size=n, replace=True)

        # Choose k random data points for initial centers
        centers = np.array([X[i] for i in np.random.choice(range(X.shape[0]),
                                                           size=self.k)])

        i = 0
        while i < self.max_iter:

            # For each row (example), find its nearest cluster
            for xi, x in enumerate(X):
                #               0              1
                result = min([(distance(x, c), ci) for ci, c in enumerate(centers)])
                labels[xi] = result[1]

            # Move centroids
            # Update the location of each centroid to the center of corresponding
            # points
            for ci, c in enumerate(centers):
                X_subset = X[labels == ci, :]
                centers[ci, :] = X_subset.mean(axis=0)

            i = i + 1

        return labels, centers




from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import dendrogram, linkage

import pandas as pd
import numpy as np
from scipy import stats, integrate
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

dfRatings = pd.read_csv('../data/ratings.csv')
dfMovies = pd.read_csv('../data/movies.csv')

dfMovies['genres'] = dfMovies['genres'].apply(lambda x: x.split("|")[0])


df = dfRatings.pivot(index='movieId', columns='userId', values='rating')
df['popularity'] = len(df.columns) - df.isnull().sum(axis=1)
df.sort_values(by='popularity', axis=0, ascending=False, inplace=True)
#del df['popularity']
df = df[:100]

#df = df.dropna(axis=0, how='any', thresh=120, subset=None, inplace=False)
#
# df = df.fillna(0)
df = df.join(dfMovies)
print(df.groupby(by='genres').count().sort_values(by="title", ascending=False))
writer = pd.ExcelWriter('output.xlsx')
df.to_excel(writer,'Sheet1')
writer.save()


# Testirajte razred KMeans


# Trenutno so gruče dodeljene naključno

k = 5

model = KMeans(k=k, max_iter=50)
labels, centers = model.fit(df.as_matrix())

plt.figure(figsize=(10, 10))
color = {0:"red", 1:"blue", 2:"yellow"}

# print(labels)
#
# print(df)

mv = dfMovies[dfMovies['movieId'].isin(df.index)]

mv['label'] = labels

for i in range(k):
    print(len(mv[mv['label'].isin([i,])]))




# fill missing values with mean value of movie rating
# df = df.T
# df = df.fillna(df.mean())
# df = df.T
# df = df.round(0).apply(np.int64)


# X = df.as_matrix()
# X = StandardScaler().fit_transform(X)
# db = DBSCAN(eps=0.3, min_samples=1).fit(X)
# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True
# labels = db.labels_
#
# print(labels)
#
# # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#
# print('Estimated number of clusters: %d' % n_clusters_)
#
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(X, labels))
X = df.as_matrix()
Z = linkage(X, 'ward')

print(Z)

plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=10,  # font size for the x axis labels
    truncate_mode='lastp',  # show only the last p merged clusters
    p=10,
)
plt.savefig('myfig4.png')

