import numpy as np
import numpy.typing as npt
from typing import Tuple
import matplotlib.pyplot as plt

import random
import time

def getPolylines(n : int= random.randint(4, 10), m=0) -> Tuple[npt.NDArray, npt.NDArray] :
    if not m :
        m += n

    pl1 : npt.NDArray = np.random.rand(n,2).round(4)
    pl2 : npt.NDArray = np.random.rand(m,2).round(4)

    return pl1, pl2

def getSinusoidPolylines(n : int= random.randint(4, 10), m=0) -> Tuple[npt.NDArray, npt.NDArray] :
    if not m :
        m += n
    A = random.random() * 10
    w = random.random() * 6.28

    pl1 : npt.NDArray = np.array([[i, A*np.sin(i*w) + random.random()] for i in range(1,n)]).round(4)
    pl2 : npt.NDArray = np.array([[i, A*np.sin(i*w) + random.random()]for i in range(1,m)]).round(4)

    return pl1, pl2

def show(pl1: npt.NDArray, pl2: npt.NDArray, title: str = "") :
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