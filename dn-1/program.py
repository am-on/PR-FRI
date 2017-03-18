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

    genres = sorted(genres.items(), key=lambda x: x[1])[::-1]

    x = range(len(genres))

    plt.figure(figsize=(20, 15))
    plt.bar(x, [n for genre, n in genres])
    plt.xlim(-0.5, len(genres) - 0.5)
    plt.xticks(x)
    plt.gca().set_xticklabels([genre for genre, n in genres], rotation=90)

    plt.ylabel("Število filmov")

    plt.show()
    plt.savefig('myfig.png')


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

    plt.figure(figsize=(20, 15))
    plt.plot([score for score, n, move in r], [n for score, n, move in r])

    plt.show()
    plt.savefig('myfig2.png')

    return r

def ratingsThroughTime(timeFrame):
    reader = DictReader(open('data/ratings.csv', 'rt', encoding='utf-8'))

    movieRatings = dict()

    # read data
    for row in reader:
        user = row['userId']
        movie = row['movieId']
        rating = row['rating']
        timestamp = int(row['timestamp'])

        if movie not in movieRatings.keys():
            movieRatings[movie] = []

        # last float(rating) is reserved for mean rating of last timeFrame ratings
        movieRatings[movie] = movieRatings[movie] + [(timestamp, float(rating), float(rating))]

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
    plt.figure(figsize=(20, 15))
    plt.hist(meanDist, normed=False, bins=15)
    plt.xlabel("X - Razlika v povprečju %s ocen" % timeFrame)
    plt.ylabel("Število vzorcev")
    plt.savefig('myfig3.png')

    n = len(meanDist)
    mu = np.mean(meanDist)  # ocena sredine
    sigma2 = (n - 1) / n * np.var(meanDist)  # ocena variance

    # Meritev, ki bi jo radi statisticno ocenili
    qx = 0.9

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
    plt.plot(xr, Px, linewidth=2.0, label="aa" )
    plt.fill_between(ltx, 0, P_ltx, alpha=0.2, )
    plt.text(qx, mvn.pdf(qx, mu, sigma2) * width,
             "p=%f" % p_value,
             horizontalalignment="right",
             verticalalignment="center", )

    plt.xlabel("X - povprečna ocena šale.")
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
    print("Verjetnost šale z oceno %.3f ali manj (statistična značilnost): " % qx + "%.3f" % (100 * p_value) + " %")
    print("Nenavadnost šale %s statistično značilna (prag = %.3f" % (sig, 100 * alpha), "%)")


    for movie, dist in dist:
        if abs(dist) > 0.8 and len(movieRatings[movie]) > 100:
            #print(movieRatings[movie])
            x = []
            y = []

            for date, rate, meanR in movieRatings[movie]:
                x.append(date)
                y.append(meanR)
            print(x)
            print(y)
            plt.figure(figsize=(20, 15))

            plt.plot(range(len(y[2:])), y[2:])
            plt.axis([0, len(y[2:]), 0, 5])
            plt.show()
            plt.savefig(movie + '.png')


    #print(movieRatings['60'])


ratingsThroughTime(30)

genresDist()

for score, movie in bestMeanRatedMovies()[0:10]:
    print(round(score,3), movie)
