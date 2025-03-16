import open3d as o3d
import numpy as np
import random
from sklearn.neighbors import KDTree

def split_kd_tree(points, max_points_per_block):
    tree = KDTree(points)
    leaf_indices = []

    def recursive_split(indices):
        if len(indices) <= max_points_per_block:
            leaf_indices.append(indices)
            return

        # Find the dimension with the largest variance
        data = points[indices]
        dim = np.argmax(np.var(data, axis=0))

        # Split the data along the median of the chosen dimension
        median = np.median(data[:, dim])
        left_indices = indices[data[:, dim] <= median]
        right_indices = indices[data[:, dim] > median]

        if len(left_indices) > 0:
            recursive_split(left_indices)
        if len(right_indices) > 0:
            recursive_split(right_indices)

    all_indices = np.arange(points.shape[0])
    recursive_split(all_indices)

    return leaf_indices

def farthest_point_sampling(points, k):
    N, D = points.shape
    centroids = np.zeros((k,), dtype=int)
    distances = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(k):
        centroids[i] = farthest
        centroid = points[farthest, :]
        dist = np.sum((points - centroid) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        farthest = np.argmax(distances)
    return centroids