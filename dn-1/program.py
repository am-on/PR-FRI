from csv import DictReader
import numpy as np

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

for score, movie in bestMeanRatedMovies()[0:10]:
    print(round(score,3), movie)
