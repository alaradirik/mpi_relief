import time
import csv
import random
import argparse
import numpy as np


def get_input_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", dest="path", help="input filepath", required=True)

    args = parser.parse_args()
    return args


def read_tsv(path):
    """
    Reads input tsv file, return data and training configuration.

    Parameters:
        path (str): path to input tsv file

    Returns:
        data (2D numpy array (n, # of features + 1)): data array.
        config (1D numpy array (1, 5)): # of processes p, # of training
        instances n, # of iterations m, # of features to retain t.
    """
    tsv_file = open(path)
    read_tsv = csv.reader(tsv_file, delimiter="\t")

    data, config = [], []
    i = 0

    for row in read_tsv:
        if i == 0:
            config.append(int(row[0]))
        elif i == 1:
            config.extend([int(x) for x in row])
        else:
            data.append([float(x) for x in row])
        i += 1

    tsv_file.close()
    return np.asarray(data), np.asarray(config)


def read_txt(path):
    """
    Read input txt file, return data and training configuration.

    Parameters:
        path (str): path to input txt file

    Returns:
        data (2D numpy array (n, # of features + 1)): data array.
        config (1D numpy array (1, 5)): # of processes p, # of training
        instances n, # of iterations m, # of features to retain t.
    """
    lines = [line.rstrip() for line in open(path)]
    setup = [int(x) for x in lines[1].split(" ")]
    config = [int(lines[0]), setup[0], setup[1], setup[2], setup[3]]
    data = [[float(x) for x in line.split(" ")] for line in lines[2:]]
    return np.asarray(data), np.asarray(config)


def read_input(path):
    """
    Reads input file, returns data and training configuration.
    """
    if path.endswith(".tsv"):
        data, config = read_tsv(path)
    else:
        data, config = read_txt(path)

    return data, config


def diff(A, X, Y, A_rng):
    """
    Return the normalized difference between the Ath feature
    of data instances X and Y.

    Parameters:
        A (int): index of the feature variable to be considered.
        X (1D numpy array (1, # of features)): data instance.
        Y (1D numpy array (1, # of features)): data instance (hit or miss).
        A_rng (float): difference between the max and min values of Ath
        feature in the data partition.

    Returns:
        result (float): normalized difference between the Ath feature of
        data instances X and Y.
    """
    result = abs(X[A] - Y[A]) / (A_rng)
    return result


def get_dist(Ri, X):
    """
    Returns the Manhattan distance between instances Ri and X.

    Parameters:
        Ri (1D numpy array (1, # of features)): data instance.
        X (1D numpy array (1, # of features)): data instance.

    Returns:
        dist (float): the Manhattan distance between data
        instances Ri and X.
    """
    dist = np.sum(np.abs(Ri - X), axis=0)
    return dist


def find_hit_miss(Ri, data, i):
    """
    Returns the nearest data instances of the same and opposite classes

    Parameters:
        Ri (1D numpy array (1, # of features + 1)): data instance.
        data (2D numpy array (n, # of features + 1)): data partition.
        i (int): index of instance Ri.

    Returns:
        top_t (numpy array (1, t)): top t features with the highest
        absolute weights.
    """
    hit_dist, miss_dist = np.inf, np.inf
    hit_ind, miss_ind = -1, -1

    n, a = data.shape[0], data.shape[1] - 1

    for j in range(n):
        if j == i:
            continue

        # Get the Manhattan distance between instances
        dist = get_dist(Ri[:a], data[j, :a])

        # Update the nearest instance with the same class
        if (Ri[a] == data[j, a]) and dist < hit_dist:
            hit_dist = dist
            hit_ind = j

        # Update the nearest instance with the opposite class
        if (Ri[a] != data[j, a]) and dist < miss_dist:
            miss_dist = dist
            miss_ind = j

    return hit_ind, miss_ind


def run_relief(data):
    """
    Runs the Relief algorithm on the data partition and returns
    the top t features.

    Parameters:
        data (2D numpy array (n, # of features + 1)): data partition
        assigned to the processor.

    Returns:
        hit_ind (int): index of the hit instance.
        miss_ind (int): index of the miss instance.
    """

    # Number of training instances
    n = data.shape[0] - 1
    # Number of features
    a = data.shape[1] - 1
    # Number of iterations
    m = int(data[-1, -2])
    # Number of features to retain
    t = int(data[-1, -1])

    # Remove the last row (containing m and t)
    data = data[:-1, :]

    # Initialize feature weights with 0s
    W = np.zeros((a))

    # Do m iterations
    for k in range(m):
        # Select a target instance Ri
        i = random.choice(range(n))
        Ri = data[i]

        # Find nearest hit H and nearest miss M
        hit_ind, miss_ind = find_hit_miss(Ri, data, i)
        H, M = data[hit_ind], data[miss_ind]

        # Update weights
        for j in range(a):
            # Get min - max values of jth feature
            A_min = np.min(data, axis=0)[j]
            A_max = np.max(data, axis=0)[j]
            A_rng = A_max - A_min

            W[j] = W[j] - (diff(j, Ri, H, A_rng) / m) + (diff(j, Ri, M, A_rng) / m)

    idx = (np.abs(W)).argsort()[-t:]
    top_t = np.sort(idx)
    return top_t
