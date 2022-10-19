from utils import getSinusoidPolylines, getPolylines
from compare import compareResults, compareTimes, compareTimesOnce

def main() :
    # pl1, pl2 = getPolylines(n=10, m=12) #pl1: n elements, pl2 : m elements
    pl1, pl2 = getSinusoidPolylines() #also has optional n & m arguments

    compareResults(pl1, pl2)
    compareTimesOnce(pl1, pl2)
    compareTimes(1000)

if __name__ == "__main__":
    main()
