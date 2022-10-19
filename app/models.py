import numpy as np
import numpy.typing as npt
from typing import Tuple

from scipy.spatial.distance import euclidean

from frechetdist import frdist
from fastdtw import fastdtw


# LIBRARY DISTANCES
def libFrechetDistance(pl1: npt.NDArray, pl2: npt.NDArray) -> float :
    return frdist(pl1, pl2)

def libDTWDistance(pl1: npt.NDArray, pl2: npt.NDArray) -> float :
    return fastdtw(pl1, pl2, dist=euclidean)[0]/max(len(pl1), len(pl2))


# FRECHET DISTANCE
def euclideanDistance(p1: npt.NDArray, p2: npt.NDArray) -> float :
    return np.sqrt(np.power(p1[0] - p2[0], 2) + np.power(p1[1] - p2[1], 2))

def getDistanceMatrix(pl1: npt.NDArray, pl2: npt.NDArray) -> npt.NDArray :
    distanceMatrix : npt.NDArray = np.ones((pl1.shape[0], pl2.shape[0]), dtype=float) * -1.0

    for i,p1 in enumerate(pl1) :
        for j,p2 in enumerate(pl2) :
            distanceMatrix[i,j] = euclideanDistance(p1, p2)

    return distanceMatrix

def frechetDistance(pl1 : npt.NDArray, pl2 : npt.NDArray) -> float :
    l1 : int = pl1.shape[0]
    l2 : int = pl2.shape[0]

    # Init Euclidean distance matrix
    dm : npt.NDArray = getDistanceMatrix(pl1, pl2)

    # Init Frechet Matrix
    fm : npt.NDArray = np.ones((l1, l2), dtype=float) * -1.0
    fm[0,0] = dm[0,0]

    # Load the first column and first row with distances
    for i in range(1, l1):
        fm[i, 0] = max(fm[i-1, 0], dm[i, 0])
    for j in range(1, l2):
        fm[0, j] = max(fm[0, j-1], dm[0, j])

    # Compute Frechet Matrix
    for i in range(1, l1):
        for j in range(1, l2):
            fm[i, j] = max(dm[i, j] , min(fm[i-1, j], fm[i, j-1], fm[i-1, j-1]) )

    return fm[l1-1, l2-1]


# DYNAMIC WARPING DISTANCE
def initDynamicTimeWarpingMatrix(dm : npt.NDArray, l1 : int, l2 : int) -> npt.NDArray :
    # Init Dynamic Time Warping Matrix
    dtwm : npt.NDArray = np.ones((l1, l2), dtype=float) * np.inf
    dtwm[0,0] = dm[0,0]

    # Load the first column and first row with distances
    for i in range(1, l1):
        dtwm[i, 0] = dtwm[i-1, 0] + dm[i, 0]
    for j in range(1, l2):
        dtwm[0, j] = dtwm[0, j-1] + dm[0, j]

    return dtwm

def dynamicTimeWarpingDistance(pl1 : npt.NDArray, pl2 : npt.NDArray) -> float :
    l1 : int = pl1.shape[0]
    l2 : int = pl2.shape[0]

    # Init Euclidean distance matrix
    dm : npt.NDArray = getDistanceMatrix(pl1, pl2)

    # Init Dynamic Time Warping Matrix
    dtwm : npt.NDArray = initDynamicTimeWarpingMatrix(dm, l1, l2)

    # Compute Dynamic Time Warping Matrix
    for i in range(1, l1):
        for j in range(1, l2):
            dtwm[i, j] = dm[i, j] + min(dtwm[i-1, j], dtwm[i, j-1], dtwm[i-1, j-1])

    return dtwm[l1-1, l2-1]/max(l1,l2)


#K-DYNAMIC WARPING DISTANCE
def initKDynamicTimeWarpingMatrix(dm : npt.NDArray, dtwm : npt.NDArray, l1 : int, l2 : int) -> npt.NDArray :
    # Init k-Dynamic Time Warping matrix 
    kdtwm : npt.NDArray = np.ones((l1, l2, 3), dtype=float) * -1

    # Load the first column and first row with 3 biggest pairs
    for i in range(2, l1):
        minIdx = np.argmin(dtwm[:i+1,0])
        kdtwm[i, 0] = np.sort(dm[:minIdx+1,0])[-3:]
    for j in range(2, l2):
        minIdx = np.argmin(dtwm[0,:j+1])
        kdtwm[0, j] = np.sort(dm[0,:minIdx+1])[-3:]

    return kdtwm

def init_1_1(dm : npt.NDArray, dtwm : npt.NDArray, kdtwm : npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray] :
    # Init dtwm[1,1]
    tmpArray = np.array([dtwm[0, 0], dtwm[1, 0], dtwm[0, 1]])
    minIdx = np.argmin(tmpArray)
    dtwm[1, 1] = dm[1, 1] + tmpArray[minIdx]

    # Init kdtwm[1,1]
    if (minIdx == 0) :
        kdtwm[1, 1] = np.sort(np.array([-1, dm[0, 0], dm[1, 1]]))
    elif (minIdx == 1) :
        kdtwm[1, 1] = np.sort(np.array([dm[0, 0], dm[1, 0], dm[1, 1]]))
    else :
        kdtwm[1, 1] = np.sort(np.array([dm[0, 0], dm[0, 1], dm[1, 1]]))
    
    return dtwm, kdtwm

def initSecondLineCol(dm : npt.NDArray, dtwm : npt.NDArray, kdtwm : npt.NDArray, l1 : int, l2 : int) -> Tuple[npt.NDArray, npt.NDArray] :
    # Init dtwm[1, 2] & kdtwm[1, 2]
    if (dtwm[0, 2] < dtwm[1, 1]) :
        dtwm[1, 2] = dtwm[0, 2] + dm[1, 2]
        kdtwm[1, 2] = np.sort(np.concatenate((np.array([dm[1, 2]]), kdtwm[0, 2])))[-3:]
    else :
        dtwm[1, 2] = dtwm[1, 1] + dm[1, 2]
        kdtwm[1, 2] = np.sort(np.concatenate((np.array([dm[1, 2]]), kdtwm[1, 1])))[-3:]

    # Init dtwm[2, 1] & kdtwm[2, 1]
    if (dtwm[2, 0] < dtwm[1, 1]) :
        dtwm[2, 1] = dtwm[2, 0] + dm[2, 1]
        kdtwm[2, 1] = np.sort(np.concatenate((np.array([dm[2, 1]]), kdtwm[2, 0])))[-3:]
    else :
        dtwm[2, 1] = dtwm[1, 1] + dm[2, 1]
        kdtwm[2, 1] = np.sort(np.concatenate((np.array([dm[2, 1]]), kdtwm[1, 1])))[-3:]

    # Init dtwm & kdtwm second line
    for i in range(3, l1):
        tmpArray = np.array([dtwm[i-1, 0], dtwm[i-1, 1], dtwm[i, 0]])
        minIdx = np.argmin(tmpArray)
        dtwm[i, 1] = dm[i, 1] + tmpArray[minIdx]
        kdtwm[i, 1] = np.sort(np.concatenate((np.array([dm[i, 1]]), [kdtwm[i-1, 0], kdtwm[i-1, 1], kdtwm[i, 0]][minIdx])))[-3:]
    # Init dtwm & kdtwm second column
    for j in range(3, l2):
        tmpArray = np.array([dtwm[0, j-1], dtwm[0, j], dtwm[1, j-1]])
        minIdx = np.argmin(tmpArray)
        dtwm[1, j] = dm[1, j] + tmpArray[minIdx]
        kdtwm[1, j] = np.sort(np.concatenate((np.array([dm[1, j]]), [kdtwm[0, j-1], kdtwm[0, j], kdtwm[1, j-1]][minIdx])))[-3:]

    return dtwm, kdtwm

# Actually 3-Dynamic Time Warping Distance
def kDynamicTimeWarpingDistance(pl1 : npt.NDArray, pl2 : npt.NDArray) -> float :
    l1 : int = pl1.shape[0]
    l2 : int = pl2.shape[0]

    # Init Euclidean distance matrix
    dm : npt.NDArray = getDistanceMatrix(pl1, pl2)

    # Init Dynamic Time Warping Matrix
    dtwm : npt.NDArray = initDynamicTimeWarpingMatrix(dm, l1, l2)

    # Init k-Dynamic Time Warping matrix 
    kdtwm : npt.NDArray = initKDynamicTimeWarpingMatrix(dm, dtwm, l1, l2)

    # Init [1,1]
    dtwm, kdtwm = init_1_1(dm, dtwm, kdtwm)

    # Init second line & column
    dtwm, kdtwm = initSecondLineCol(dm, dtwm, kdtwm, l1, l2)

    # Compute Matrixs
    for i in range(2, l1):
        for j in range(2, l2):
            tmpArray = np.array([dtwm[i-1, j-1], dtwm[i-1, j], dtwm[i, j-1]])
            minIdx = np.argmin(tmpArray)
            dtwm[i, j] = dm[i, j] + tmpArray[minIdx]
            kdtwm[i, j] = np.sort(np.concatenate((np.array([dm[i, j]]), [kdtwm[i-1, j-1], kdtwm[i-1, j], kdtwm[i, j-1]][minIdx])))[-3:]

    return np.mean(kdtwm[-1, -1])