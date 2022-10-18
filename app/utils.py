import numpy as np
import numpy.typing as npt
from typing import Tuple
import matplotlib.pyplot as plt

import random
import time

def euclideanDistance(p1: npt.NDArray, p2: npt.NDArray) -> float :
    return np.sqrt(np.power(p1[0] - p2[0], 2) + np.power(p1[1] - p2[1], 2))

def getDistanceMatrix(pl1: npt.NDArray, pl2: npt.NDArray) -> npt.NDArray :
    distanceMatrix : npt.NDArray = np.ones((pl1.shape[0], pl2.shape[0]), dtype=float) * -1.0

    for i,p1 in enumerate(pl1) :
        for j,p2 in enumerate(pl2) :
            distanceMatrix[i,j] = euclideanDistance(p1, p2)

    return distanceMatrix

def getPolylines() -> npt.NDArray :
    # pl1 : npt.NDArray = np.array([[0,0], [1,1], [2,2], [3,3], [4,4]])
    # pl2 : npt.NDArray = np.array([[0,0], [1,2], [2,3], [3,1], [4,4]])
    n = random.randint(4, 10)

    pl1 : npt.NDArray = np.random.rand(n,2).round(4)
    pl2 : npt.NDArray = np.random.rand(n,2).round(4)

    return pl1, pl2

def getSinusoidPolylines() -> Tuple[npt.NDArray] :
    n = 6#random.randint(4, 10)
    A = random.random() * 10
    w = random.random() * 6.28

    pl1 : npt.NDArray = np.array([[i, A*np.sin(i*w) + random.random()] for i in range(1,n)]).round(4)
    pl2 : npt.NDArray = np.array([[i, A*np.sin(i*w) + random.random()]for i in range(1,n)]).round(4)
    pl3 : npt.NDArray = np.array([[i, A*np.sin(i*w) + (random.random()*3)]for i in range(1,n)]).round(4)

    return pl1, pl2, pl3

def show(pl1: npt.NDArray, pl2: npt.NDArray, title: int = 0) :
    xpl1 = [p for p,_ in pl1]
    ypl1 = [p for _,p in pl1]
    plt.plot(xpl1, ypl1)
    plt.xlabel('X axis')

    xpl2 = [p for p,_ in pl2]
    ypl2 = [p for _,p in pl2]
    plt.plot(xpl2, ypl2)
    plt.ylabel('Y axis')

    if title :
        plt.title(f"Result : {title}")
    
    plt.show()


def display(title : str, r1 : float, r2 : float) :
    print(f"\n{title} :")
    print("Handmade : ", r1)
    print("Library : ", r2)
    if (r1-r2) :
        print(f"Difference : {100*abs(r1-r2)/r2}%")

def testTime(func, pl1 : npt.NDArray, pl2 : npt.NDArray, message : str = "") -> float :
    if message :
        print(message)
    beg = time.time()
    result = func(pl1, pl2)
    print(f"exec time : {(time.time()-beg)*1000:10.5f} ms.")
    return result

def getTime(func, pl1 : npt.NDArray, pl2 : npt.NDArray) -> float :
    beg = time.time()
    func(pl1, pl2)
    return time.time()-beg