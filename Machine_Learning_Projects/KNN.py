from numpy import *

def euclidean_distance(x, y):
    return sum((xi - yi) ** 2 for xi, yi in zip(x, y)) ** 0.5

def knn(data, query, k, distance_fn, choice_fn):
    distances = [(distance_fn(datum, query), i) for i, datum in enumerate(data)]
    sorted_distances = sorted(distances)
    k_nearest = sorted_distances[:k]
    return k_nearest, [data[i] for _, i in k_nearest]