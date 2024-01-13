import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy
from scipy.stats import multivariate_normal

import multiprocessing as mp

def line_line_distance(seg1, seg2):
    """
    Calculate the shortest distance between two line segments in 3D.

    Parameters:
    seg1: Tuple of numpy arrays representing the start and end points of the first line segment.
    seg2: Tuple of numpy arrays representing the start and end points of the second line segment.

    Reference: 
    https://www.geometrictools.com/Documentation/DistanceLine3Line3.pdf
    https://stackoverflow.com/questions/38637542/finding-the-shortest-distance-between-two-3d-line-segments
    """
    p1, q1 = seg1
    p2, q2 = seg2

    # Direction vectors for the line segments
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2

    # Squared magnitudes of the direction vectors
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, r)
    e = np.dot(d2, r)
    f = a * c - b * b

    # Check if the lines are parallel
    if f != 0:
        s = (b * e - c * d) / f
        t = (a * e - b * d) / f
    else:
        # Lines are parallel, use arbitrary point on seg1
        s = 0
        t = d / b if b > 0 else 0

    # Cap s and t between 0 and 1 to stay within the line segments
    s = max(0, min(1, s))
    t = max(0, min(1, t))

    # Calculate the closest points
    closest_point_on_seg1 = p1 + d1 * s
    closest_point_on_seg2 = p2 + d2 * t

    # Return the distance between the closest points
    return np.linalg.norm(closest_point_on_seg1 - closest_point_on_seg2)

def generate_random_sticks(n, length, x_range, y_range, z_range):
    """
    Generate a set of random sticks (line segments) in 3D space.

    Each stick is represented by its start and end points.

    Parameters:
    n : int
        The number of sticks to generate.
    length : float
        The length of each stick.
    x_range : tuple
        The range (min, max) for the x-coordinate of the starting point.
    y_range : tuple
        The range (min, max) for the y-coordinate of the starting point.
    z_range : tuple
        The range (min, max) for the z-coordinate of the starting point.

    Returns:
    sticks : numpy.ndarray
        An array of shape (n, 2, 3), where each row represents a stick with its start and end points.
    """
    sticks = np.zeros((n, 2, 3))

    for i in range(n):
        # Generate a random start point within the specified ranges
        start = np.array([np.random.uniform(x_range[0], x_range[1]),
                          np.random.uniform(y_range[0], y_range[1]),
                          np.random.uniform(z_range[0], z_range[1])])
        
        # Generate a random direction vector and normalize it
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)

        # Scale the direction vector to the specified length and calculate the end point
        end = start + direction * length

        # Store the start and end points of the stick
        sticks[i] = [start, end]

    return sticks

def parallel_pairwise_distances(sticks):
    num_sticks = len(sticks)
    pair_distance = np.zeros((num_sticks, num_sticks))

    # Create a list of all pairs of sticks for which we want to calculate distances
    pairs = [(sticks[i], sticks[j]) for i in range(num_sticks) for j in range(num_sticks) if i != j]

    # Use multiprocessing to compute distances in parallel
    with mp.Pool(mp.cpu_count()) as pool:
        distances = pool.starmap(line_line_distance, pairs)

    # Fill in the pairwise distance matrix
    k = 0
    for i in range(num_sticks):
        for j in range(num_sticks):
            if i != j:
                pair_distance[i][j] = distances[k]
                k += 1

    return pair_distance

def compute_mulnormal(index, mean_range, diag_cov, sample_space, shift):
    dend_mean = np.random.uniform(mean_range[0], mean_range[1], 3)
    dend_dist = multivariate_normal(dend_mean, diag_cov)
    opt_shift = np.random.uniform(-shift, shift, 3)
    axon_mean = np.add(dend_mean, opt_shift)
    axon_dist = multivariate_normal(axon_mean, diag_cov)

    pdf1 = dend_dist.pdf(sample_space)
    pdf2 = axon_dist.pdf(sample_space)
    return pdf1, pdf2

def generate_equal_length_stick_structure(mean_range, short_std, long_std):
    """
    Generate points for a stick-like structure with a random orientation and equal length using multivariate normal distribution.
    """
    # Creating a random rotation matrix for the covariance
    random_rotation = scipy.linalg.orth(np.random.rand(3, 3))

    # Fixed covariance along one axis (to create the stick-like shape), then rotating it
    # The variance along the major axis is set to a fixed value to maintain equal lengths
    base_cov = np.diag([short_std, short_std, long_std])  # Smaller variance in X and Y to keep the stick thin
    rotated_cov = random_rotation @ base_cov @ random_rotation.T

    return multivariate_normal(mean_range, rotated_cov)