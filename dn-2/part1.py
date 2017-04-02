
import pandas as pd
import numpy as np
from scipy import stats, integrate
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

def gausParams(x):
    mu_fit = np.mean(x)
    sigma2_fit = (len(x) - 1) / len(x) * np.var(x)
    return (mu_fit, sigma2_fit)

dfRatings = pd.read_csv('../data/ratings.csv')
dfMovies = pd.read_csv('../data/movies.csv')


def pTest(qx, data, graph=False):
    n = len(data)
    mu = data.mean() # ocena sredine
    sigma2 = (n - 1) / n * data.var()  # ocena variance


    # Izračunamo P(x) za dovolj velik interval
    xr = np.linspace(0, 5, 250)
    width = xr[1] - xr[0]  # sirina intervala
    Px = [mvn.pdf(x, mu, sigma2) * (xr[1] - xr[0]) for x in xr]


    # Vse vrednosti, ki so manjše od qx
    ltx = xr[xr > qx]

    # Množimo s širino intervala, da dobimo ploščino pod krivuljo
    P_ltx = [mvn.pdf(x, mu, sigma2) * width for x in ltx]

    # p-vrednost: ploscina pod krivuljo P(x) za vse vrednosti, manjse od qx
    p_value = np.sum(P_ltx)

    if graph:
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

    return p_value < alpha


# print(dfRatings.groupby(by='movieId')['rating'].var())
#
# print(dfRatings.loc[dfRatings['movieId'] == 163949])

# transform df
df = dfRatings.pivot(index='userId', columns='movieId', values='rating')
df = df.dropna(axis=1, how='any', thresh=20, subset=None, inplace=False)


# fill missing values with mean value of movie rating
df = df.T
df = df.fillna(df.mean())
df = df.T
# df = df.round(0).apply(np.int64)

sns.distplot(df[260]).figure.savefig("output.png")

fig = sns.distplot(df.var(), kde=False, hist=False, fit=stats.nct, fit_kws={"label": "studentova porazdelitev", "color":"r"} )
fig = sns.distplot(df.var(), fit=stats.nct, kde_kws={"label": "dejanska porazdelitev"}, fit_kws={"label": "normalna porazdelitev", "color":"r"}, color='b' )


plt.yticks(fig.get_yticks(), fig.get_yticks() * 100)
plt.xlim(0,5)
fig.set(xlabel='varianca', ylabel='število filmov')
plt.title('Porazdelitev variance ocen filmov')
fig.figure.savefig("porazdelitev.png")

plt.figure(figsize=(10, 5))
plt.suptitle('Odvisnost med številom ogledov in povprečno oceno filma', fontsize=14, fontweight='bold')
plt.hist(df.var().get_values())
plt.xlabel("Povprečna ocena")
plt.ylabel("Število oglecov (ocen)")

plt.show()
plt.savefig('viewsVsMeanRate.png')

# print(df)
print(df.var().nlargest(10))

x = df.var().get_values()
# print(gausParams(x))

# unordinary = [pTest(v,x) for v in x]
#
#
#
# print(df.var()[unordinary])
#
# pTest(1.379,x, graph=True)
stat = list(df.var().nlargest(10).index)
print(dfMovies.loc[dfMovies['movieId'].isin(stat)])
