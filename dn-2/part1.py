
import pandas as pd
import numpy as np
from scipy import stats, integrate
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

def pTest(qx, data, graph=False):
    n = len(data)
    mu = data.mean() # ocena sredine
    sigma2 = (n - 1) / n * data.var()  # ocena variance

    params = stats.beta.fit(data)


    # Izračunamo P(x) za dovolj velik interval
    xr = np.linspace(0, 5, 250)
    width = xr[1] - xr[0]  # sirina intervala

    Px = [stats.beta.pdf(x, *params) * (xr[1] - xr[0]) for x in xr]


    # Vse vrednosti, ki so manjše od qx
    ltx = xr[xr > qx]

    # Množimo s širino intervala, da dobimo ploščino pod krivuljo
    P_ltx = [ stats.beta.pdf(x, *params) * width for x in ltx]

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



dfRatings = pd.read_csv('../data/ratings.csv')
dfMovies = pd.read_csv('../data/movies.csv')

# transform df
df = dfRatings.pivot(index='userId', columns='movieId', values='rating')
df = df.dropna(axis=1, how='any', thresh=10, subset=None, inplace=False)


x = df.var()

fig = sns.distplot(x, kde=False, hist=False, fit=stats.beta, fit_kws={"label": "beta porazdelitev", "color":"r"} )
fig = sns.distplot(x, fit=stats.beta, kde_kws={"label": "dejanska porazdelitev"}, fit_kws={"label": "beta porazdelitev", "color":"r"}, color='b' )
#
#
plt.yticks(fig.get_yticks(), fig.get_yticks() * 100)
plt.xlim(0,4)
fig.set(xlabel='varianca', ylabel='število filmov')
plt.title('Porazdelitev varianc ocen filmov')
fig.figure.savefig("porazdelitevBeta.png")





params = stats.beta.fit(x)
print(params)

unordinary = [pTest(v,x) for v in x]

print(len(unordinary))

print(list(df.var().nlargest(10)))

print(dfMovies[dfMovies['movieId'].isin(x[unordinary].nlargest(10).index)]['title'])
print(len(dfMovies[dfMovies['movieId'].isin(x[unordinary].index)]['title']))