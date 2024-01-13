from sklearn.manifold import Isomap
from kneed import KneeLocator

# isomap analysis is time-consuming when sample size is large
def isomap_test(X, grange):
    dim = []
    re_err = []
    for n_comp in range(1, grange):
        embedding = Isomap(n_components=n_comp)
        X_transformed = embedding.fit_transform(X)
        dim.append(n_comp)
        re_err.append(embedding.reconstruction_error())
    return dim, re_err

