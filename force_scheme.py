import time
import math
import numpy as np
from numba import njit, prange



@njit(parallel=True, cache=True)
def stress(distance_matrix, projection):
    size = len(projection)
    total = len(distance_matrix)
    den = 0
    num = 0
    for i in prange(size):
        for j in range (size):
            dr2 = math.sqrt((projection[i][0] - projection[j][0]) * (projection[i][0] - projection[j][0]) +
                            (projection[i][1] - projection[j][1]) * (projection[i][1] - projection[j][1]))

            r = (i + j - math.fabs(i - j)) / 2  # min(i,j)
            s = (i + j + math.fabs(i - j)) / 2  # max(i,j)
            drn = distance_matrix[int(total - ((size - r) * (size - r + 1) / 2) + (s - r))]
            num += (drn - dr2) * (drn - dr2)
            den += drn * drn
    return math.sqrt(num / den)


@njit(parallel=True, fastmath=False, cache=True)
def move(ins1, distance_matrix, projection, learning_rate):
    size = len(projection)
    total = len(distance_matrix)
    error = 0
    
    for ins2 in prange(size):
        if ins1 != ins2:
            x1x2 = projection[ins2][0] - projection[ins1][0]
            y1y2 = projection[ins2][1] - projection[ins1][1]
            dr2 = max(math.sqrt(x1x2 * x1x2 + y1y2 * y1y2), 0.0001)

            # getting te index in the distance matrix and getting the value
            r = (ins1 + ins2 - math.fabs(ins1 - ins2)) / 2  # min(ins1,ins2)
            s = (ins1 + ins2 + math.fabs(ins1 - ins2)) / 2  # max(ins1,ins2)
            drn = distance_matrix[int(total - ((size - r) * (size - r + 1) / 2) + (s - r))]

            # calculate the movement
            delta = (drn - dr2) #* math.fabs(drn - dr2)
            
            error += math.fabs(delta)

            # moving
            projection[ins2][0] += learning_rate * delta * (x1x2 / dr2)
            projection[ins2][1] += learning_rate * delta * (y1y2 / dr2)
    return error / (size-1)

#@njit(parallel=False, fastmath=False)
def iteration(index, distance_matrix, projection, learning_rate):
    size = len(projection)
    error = 0

    for i in range(size):
        ins1 = index[i]
        error += move(ins1, distance_matrix, projection, learning_rate)
    return error / size 

@njit(parallel=False, fastmath=False, cache=True)
def iteration_no_verbose(index, distance_matrix, projection, learning_rate):
    size = len(projection)
    error = 0

    for i in prange(size):
        ins1 = index[i]
        size = len(projection)
        total = len(distance_matrix)
        error_l = 0
        for ins2 in range(size):
            if ins1 != ins2:
                x1x2 = projection[ins2][0] - projection[ins1][0]
                y1y2 = projection[ins2][1] - projection[ins1][1]
                dr2 = max(math.sqrt(x1x2 * x1x2 + y1y2 * y1y2), 0.0001)

                # getting te index in the distance matrix and getting the value
                r = (ins1 + ins2 - math.fabs(ins1 - ins2)) / 2  # min(ins1,ins2)
                s = (ins1 + ins2 + math.fabs(ins1 - ins2)) / 2  # max(ins1,ins2)
                drn = distance_matrix[int(total - ((size - r) * (size - r + 1) / 2) + (s - r))]

                # calculate the movement
                delta = (drn - dr2) #* math.fabs(drn - dr2)
                
                error_l += math.fabs(delta)

                # moving
                projection[ins2][0] += learning_rate * delta * (x1x2 / dr2)
                projection[ins2][1] += learning_rate * delta * (y1y2 / dr2)
        error += error_l / (size-1)

    return error / size 

#@njit(fastmath=False)
def execute(distance_matrix, projection, max_it, verbose, learning_rate0=0.5, lrmin= 0.05, decay=0.95, tolerance=0.0000001):
    nr_iterations = 0
    size = len(projection)
    error= math.inf
    kstress=np.zeros(max_it)
    p_error = np.zeros(max_it)
    # create random index
    index = np.random.permutation(size)
    if verbose:
        from tqdm import tqdm
        for k in tqdm(range(max_it), desc='Running ELViM projection'):
            learning_rate = max(learning_rate0 * math.pow((1 - k / max_it), decay), lrmin)
            new_error = iteration(index, distance_matrix, projection, learning_rate)
            if math.fabs(new_error - error) < tolerance:
                break
            error = new_error
            p_error[k] = error
            kstress[k] = stress(distance_matrix, projection)
            nr_iterations = k + 1 
    else:
        for k in range(max_it):
            learning_rate = max(learning_rate0 * math.pow((1 - k / max_it), decay), lrmin)
            new_error = iteration_no_verbose(index, distance_matrix, projection, learning_rate)
            if math.fabs(new_error - error) < tolerance:
                break
            error = new_error
            p_error[k] = error
            kstress[k] = stress(distance_matrix, projection)
            nr_iterations = k + 1  

    # setting the min to (0,0)
    min_x = min(projection[:, 0])
    min_y = min(projection[:, 1])
    for i in range(size):
        projection[i][0] -= min_x
        projection[i][1] -= min_y
    return nr_iterations, p_error, kstress
