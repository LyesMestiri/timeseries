import numpy.typing as npt

from scipy.spatial.distance import euclidean

from frechetdist import frdist
from fastdtw import fastdtw

from models import *
from utils import getPolylines, getSinusoidPolylines, show, testTime, getTime, display

def comparefrechet(pl1: npt.NDArray, pl2: npt.NDArray) -> float :
    return frdist(pl1, pl2)

def compareDTWDistance(pl1: npt.NDArray, pl2: npt.NDArray) -> float :
    return fastdtw(pl1, pl2, dist=euclidean)[0]/max(len(pl1), len(pl2))

def compare(pl1: npt.NDArray, pl2: npt.NDArray, func: str) -> float :
    if func == "Frechet" :
        f1, f2 = frechetDistance, frechetDistance
    elif func == "Dynamic Time Warping" :
        f1, f2 = dynamicTimeWarpingDistance, compareDTWDistance

    result = f1(pl1, pl2)
    comp = f2(pl1, pl2)

    display(func, result, comp)

def compareResults() :
    print("\tCOMPARE RESULTS :")
    pl1, pl2, pl3 = getSinusoidPolylines()

    compare(pl1, pl2, "Frechet")

    compare(pl1, pl2, "Dynamic Time Warping")

    print("\nK-Dynamic Time Warping Distance :", kDynamicTimeWarpingDistance(pl1, pl2))

    show(pl1, pl2, "Close")

    print()

def compareTimes(n: int) :
    print(f"\tCOMPARING TIMES ON {n} EXECUTIONS :\n")
    times = [0, 0, 0]
    funcs = [frechetDistance, dynamicTimeWarpingDistance, kDynamicTimeWarpingDistance]
    names = ["frechetDistance", "dynamicTimeWarpingDistance", "kDynamicTimeWarpingDistance"]

    for _ in range(n) :
        pl1, pl2, pl3 = getSinusoidPolylines()
        for i in range(3) :
            times[i] += getTime(funcs[i], pl1, pl2)

    for i in range(3) :
        print(f"{names[i]} : {(times[i]/n)*1000:10.5f} ms.")

    print()

def compareTime(func, name, pl1, pl2) :
    result = testTime(func, pl1, pl2, name)
    print(f"result : {result}\n")

def compareTimesOnce():
    print("\tCOMPARING EXECUTION TIMES :\n")
    pl1, pl2, pl3 = getSinusoidPolylines()

    compareTime(frechetDistance, "Testing Frechet Distance", pl1, pl2)

    compareTime(dynamicTimeWarpingDistance, "Testing Dynamic Time Warping Distance", pl1, pl2)

    compareTime(kDynamicTimeWarpingDistance, "Testing K-Dynamic Time Warping Distance", pl1, pl2)

    show(pl1, pl2, "Close")

    print()