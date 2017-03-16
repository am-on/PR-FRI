from csv import DictReader

import matplotlib
import numpy as np
import matplotlib.pyplot as plt



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


genresDist()

for score, movie in bestMeanRatedMovies()[0:10]:
    print(round(score,3), movie)


