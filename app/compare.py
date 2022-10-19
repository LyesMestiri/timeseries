import numpy.typing as npt

from models import *
from utils import getPolylines, getSinusoidPolylines, show, testTime, getTime, display

def compare(pl1: npt.NDArray, pl2: npt.NDArray, func: str) :
    if func == "Frechet" :
        if (len(pl1) == len(pl2)) :
            f1, f2 = frechetDistance, libFrechetDistance
        else :
            print("Can't compare Frechet")
            f1, f2 = frechetDistance, frechetDistance
    elif func == "Dynamic Time Warping" :
        f1, f2 = dynamicTimeWarpingDistance, libDTWDistance
    else :
        print(func, 'is not a valid function.')
        exit()

    result = f1(pl1, pl2)
    comp = f2(pl1, pl2)

    display(func, result, comp)

def compareResults(pl1: npt.NDArray, pl2: npt.NDArray) :
    print("\tCOMPARE RESULTS :")

    compare(pl1, pl2, "Frechet")

    compare(pl1, pl2, "Dynamic Time Warping")

    print("\nK-Dynamic Time Warping Distance :", kDynamicTimeWarpingDistance(pl1, pl2))

    show(pl1, pl2, "Close")

    print()

def compareTimes(n: int) :
    print(f"\tCOMPARING TIMES ON {n} EXECUTIONS :\n")
    times = [0.0] * 3
    funcs = [frechetDistance, dynamicTimeWarpingDistance, kDynamicTimeWarpingDistance]
    names = ["frechetDistance", "dynamicTimeWarpingDistance", "kDynamicTimeWarpingDistance"]

    for _ in range(n) :
        pl1, pl2 = getSinusoidPolylines()
        for i in range(3) :
            times[i] += getTime(funcs[i], pl1, pl2)

    for i in range(3) :
        print(f"{names[i]} : {(times[i]/n)*1000:10.5f} ms.")

    print()

def compareTime(func, name, pl1, pl2) :
    result = testTime(func, pl1, pl2, name)
    print(f"result : {result}\n")

def compareTimesOnce(pl1: npt.NDArray, pl2: npt.NDArray):
    print("\tCOMPARING EXECUTION TIMES :\n")

    compareTime(frechetDistance, "Testing Frechet Distance", pl1, pl2)

    compareTime(dynamicTimeWarpingDistance, "Testing Dynamic Time Warping Distance", pl1, pl2)

    compareTime(kDynamicTimeWarpingDistance, "Testing K-Dynamic Time Warping Distance", pl1, pl2)

    show(pl1, pl2, "Close")

    print()