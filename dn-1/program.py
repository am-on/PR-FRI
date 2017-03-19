from csv import DictReader

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from datetime import datetime


def bestMeanRatedMovies():
    # http://stackoverflow.com/questions/7089379/most-efficient-way-to-sum-huge-2d-numpy-array-grouped-by-id-column

    reader = DictReader(open('data/ratings.csv', 'rt', encoding='utf-8'))

    movieRatings = dict()

    for row in reader:
        user = row['userId']
        movie = row['movieId']
        rating = row['rating']
        timestamp = row['timestamp']

        if movie not in movieRatings.keys():
            movieRatings[movie] = []

        movieRatings[movie] = movieRatings[movie] + [float(rating),]


    reader = DictReader(open('data/movies.csv', 'rt', encoding='utf-8'))
    for row in reader:
        movie = row['movieId']
        title = row['title']

        if movie not in movieRatings.keys():
            movieRatings[movie] = []

        movieRatings[movie] = (movieRatings[movie], title)

    r = []
    for movieId, data in movieRatings.items():
        ratings, title = data
        if len(ratings) > 50:
            r.append((np.mean(ratings), title))

    r.sort(reverse=True)

    return r


def genresDist():
   
    reader = DictReader(open('data/movies.csv', 'rt', encoding='utf-8'))
    genres = dict()

    for row in reader:
        g = row['genres']
        for genre in g.split('|'):
            if(genre not in genres.keys()):
                genres[genre] = 0
            genres[genre] += 1

    print(genres)
    genres = sorted(genres.items(), key=lambda x: x[1])[::-1]
    print(genres)
    x = range(len(genres))

    plt.figure(figsize=(10, 13))
    plt.suptitle('Porazdelitev žanrov', fontsize=14, fontweight='bold')
    plt.bar(x, [n for genre, n in genres])
    plt.xlim(-0.5, len(genres) - 0.5)
    plt.xticks(x)
    plt.gca().set_xticklabels([genre for genre, n in genres], rotation=90)
    plt.ylabel("Število filmov")
    plt.show()
    plt.savefig('genresDist.png')


def numberOfRatesVsMeanRate():

    reader = DictReader(open('data/ratings.csv', 'rt', encoding='utf-8'))

    movieRatings = dict()

    for row in reader:
        user = row['userId']
        movie = row['movieId']
        rating = row['rating']
        timestamp = row['timestamp']

        if movie not in movieRatings.keys():
            movieRatings[movie] = []

        movieRatings[movie] = movieRatings[movie] + [float(rating),]

    reader = DictReader(open('data/movies.csv', 'rt', encoding='utf-8'))
    for row in reader:
        movie = row['movieId']
        title = row['title']

        if movie not in movieRatings.keys():
            movieRatings[movie] = []

        movieRatings[movie] = (movieRatings[movie], title)

    r = []

    for movieId, data in movieRatings.items():
        ratings, title = data
        if len(ratings) > 0:
            r.append((np.mean(ratings), len(ratings), title))

    r.sort()

    plt.figure(figsize=(10, 5))
    plt.suptitle('Odvisnost med številom ogledov in povprečno oceno filma', fontsize=14, fontweight='bold')
    plt.plot([score for score, n, move in r], [n for score, n, move in r])
    plt.xlabel("Povprečna ocena")
    plt.ylabel("Število oglecov (ocen)")

    plt.show()
    plt.savefig('viewsVsMeanRate.png')

    return r



def pTest(qx, data):
    n = len(data)
    mu = np.mean(data)  # ocena sredine
    sigma2 = (n - 1) / n * np.var(data)  # ocena variance


    # Izračunamo P(x) za dovolj velik interval
    xr = np.linspace(-3, 3, 35)
    width = xr[1] - xr[0]  # sirina intervala
    Px = [mvn.pdf(x, mu, sigma2) * (xr[1] - xr[0]) for x in xr]

    # Vse vrednosti, ki so manjše od qx
    ltx = xr[xr < qx]

    # Množimo s širino intervala, da dobimo ploščino pod krivuljo
    P_ltx = [mvn.pdf(x, mu, sigma2) * width for x in ltx]

    # p-vrednost: ploscina pod krivuljo P(x) za vse vrednosti, manjse od qx
    p_value = np.sum(P_ltx)

    # Graf funkcije
    plt.figure()
    plt.plot(xr, Px, linewidth=2.0, label="razlika v popularnosti med dvema obdobjema")
    plt.fill_between(ltx, 0, P_ltx, alpha=0.2, )
    plt.text(qx, mvn.pdf(qx, mu, sigma2) * width,
             "p=%f" % p_value,
             horizontalalignment="right",
             verticalalignment="center", )

    plt.xlabel("X - povprečna razlika v popularnosti med dvema obdobjema.")
    plt.ylabel("P(X)")
    plt.legend()
    plt.savefig('myfig4.png')
    plt.show()

    # Poglejmo, ali je meritev statistično značilna pri danem pragu alpha (0.05, 0.01, 0.001 ... )
    alpha = 0.05
    if p_value < alpha:
        sig = "JE"
    else:
        sig = "NI"

    # Rezultat statističnega testa
    print("Verjetnost za razliko v popularnosti %.3f ali manj (statistična značilnost): " % qx + "%.3f" % (100 * p_value) + " %")
    print("Razlika v popularnosti %s statistično značilna (prag = %.3f" % (sig, 100 * alpha), "%)")


def ratingsThroughTime(timeFrame):
    reader = DictReader(open('data/ratings.csv', 'rt', encoding='utf-8'))

    movieRatings = dict()

    # read ratings data
    for row in reader:
        user = row['userId']
        movie = row['movieId']
        rating = row['rating']
        timestamp = int(row['timestamp'])

        if movie not in movieRatings.keys():
            movieRatings[movie] = []

        # last float(rating) is reserved for mean rating of last timeFrame ratings
        movieRatings[movie] = movieRatings[movie] + [(timestamp, float(rating), float(rating))]

    #movieId, title, genres


    movieNames = dict()

    reader = DictReader(open('data/movies.csv', 'rt', encoding='utf-8'))
    for row in reader:
        id = row['movieId']
        title = row['title']

        movieNames[id] = title


    # add mean rating of last timeFrame ratings
    for movie, data in movieRatings.items():
        # sort by time
        movieRatings[movie] = sorted(movieRatings[movie], key=lambda x: x[0])


        for i, value in enumerate(movieRatings[movie]):
            if i <= 0: continue

            time, rating, meanR = value

            rsum = rating
            start = i-timeFrame if i-timeFrame >= 0 else 0

            for time, rate, meanR in movieRatings[movie][start:i]:
                rsum += rate

            movieRatings[movie][i] = (time, rating, rsum/(i-start+1))

    # get distribution of changes in mean ratings
    dist = []
    for movie, data in movieRatings.items():
        oldMeanR = data[0][2]
        for i, d in enumerate(data[1:]):
            time, rating, meanR = d
            if i % (timeFrame/2) == 0:
                dist.append((movie, oldMeanR-meanR))
                oldMeanR = meanR


    meanDist = [mean for movie, mean in dist]

    # draw hist of changes in mean ratings
    plt.figure(figsize=(8, 4))
    plt.suptitle('Razlike v popularnosti filmov skozi čas', fontsize=14, fontweight='bold')
    plt.hist(meanDist, normed=False, bins=15)
    plt.xlabel("Razlika v povprečju med dvema časovnima obdobjema (%s ocen)" % timeFrame)
    plt.ylabel("Število časovnih obdobij")
    plt.savefig('popularityChangesHist.png')

    pTest(-1, meanDist)

    for movie, dist in dist:
        #if abs(dist) > 1 and len(movieRatings[movie]) > 100:
        if movie == '595' or movie == '34':

            x = []
            y = []

            for date, rate, meanR in movieRatings[movie]:
                x.append(datetime.fromtimestamp(date).strftime("%d. %m. '%y"))
                y.append(meanR)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            fig.suptitle('Popularnost filma %s skozi čas' % movieNames[movie], fontsize=14, fontweight='bold')

            axes[0].plot(range(len(y[timeFrame:])), y[timeFrame:])
            axes[0].axis([timeFrame, len(y[timeFrame:]), 0, 5])
            axes[0].set_xlabel("Število ocen")
            axes[0].set_ylabel("Povprečna ocena v obdobju")

            axes[1].plot(range(len(y[timeFrame:])), y[timeFrame:])
            axes[1].axis([timeFrame, len(y[timeFrame:]), min(y[timeFrame:])-0.01, max(y[timeFrame:])+0.01])
            axes[1].set_xlabel("Število ocen")
            axes[1].set_ylabel("Povprečna ocena v obdobju")

            plt.show()
            plt.savefig(movie + '.png')

def castPopularity():
    reader = DictReader(open('data/ratings.csv', 'rt', encoding='utf-8'))

    movieRatings = dict()

    for row in reader:
        user = row['userId']
        movie = row['movieId']
        rating = row['rating']
        timestamp = row['timestamp']

        if movie not in movieRatings.keys():
            movieRatings[movie] = []

        movieRatings[movie] = movieRatings[movie] + [float(rating),]

    for movie, ratings in movieRatings.items():
        movieRatings[movie] = sum(movieRatings[movie])


    reader = DictReader(open('data/cast.csv', 'rt', encoding='utf-8'))

    castPopularity = dict()

    for row in reader:
        movie = row['movieId']
        cast = row['cast']

        for actor in cast.split('|'):
            if actor not in castPopularity.keys():
                castPopularity[actor] = 0
            castPopularity[actor] += movieRatings[movie] if movie in movieRatings.keys() else 0

    castPopularity = sorted(castPopularity.items(), key=lambda x: x[1])[::-1]

    return castPopularity


for actor, rang in castPopularity()[:10]:
    if actor:
        print(actor + " & " + str(rang) + " \\\\")
# numberOfRatesVsMeanRate()
#
# ratingsThroughTime(30)
# #
# genresDist()
# #
# for score, movie in bestMeanRatedMovies()[0:10]:
#     print(movie + " & " + str(round(score,3)) + " \\\\")
