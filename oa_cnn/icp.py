import numpy as np
from sklearn.neighbors import NearestNeighbors


def best_fit_transform(A, B):
    assert len(A) == len(B)

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    t = centroid_B.T - np.dot(R, centroid_A.T)

    T = np.identity(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    return T, R, t


def nearest_neighbor(src, dst):
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=200, tolerance=1e-6):
    # make points homogeneous, copy them so as to maintain the originals
    A, B = np.array(A), np.array(B)
    src = np.ones((4, A.shape[0]))
    dst = np.ones((4, B.shape[0]))
    src[0:3, :] = np.copy(A.T)
    dst[0:3, :] = np.copy(B.T)

    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbours between the current source
        # and destination points
        distances, indices = nearest_neighbor(src[0:3, :].T, dst[0:3, :].T)

        # compute the transformation between the current source
        # and nearest destination points
        T, _, _ = best_fit_transform(src[0:3, :].T, dst[0:3, indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.sum(distances) / distances.size
        if abs(prev_error-mean_error) < tolerance:
            # print(f'tolerence atteind after {i} iterations')
            break
        prev_error = mean_error

    # print(f'maximum distance after icp : {max(distances)}')
    T, _, _ = best_fit_transform(A, src[0:3, :].T)

    return T
