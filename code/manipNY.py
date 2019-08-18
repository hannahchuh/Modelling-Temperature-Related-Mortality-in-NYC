"""
    Take in date and min/mean/max temp data for 1973-2015. Take in mortality
    data for 1987-2000. Make smoothed lists, de-trended mortality, heat wave
    percentile graphs, multiple percentile graphs, etc.
"""

# LIBRARIES
import pickle
import matplotlib.pyplot as plt
import numpy as np
import csv
from decimal import Decimal
from scipy import stats
from scipy import optimize as opt
from decimal import ROUND_CEILING
import matplotlib.patches as mpatches
from sklearn import linear_model

# FUNCTIONS

def tempInRange( temp, threshold, percent ):
    """
    Check if temperature is within threshold; check for above/below
    threshod depending on percent
    :param temp: float/decimal
    :param threshold: float/decimal
    :param percent: float/decimal
    :return: boolean
    """

    if percent < .50:
        if temp < threshold:
            return True
        else:
            return False

    if percent >= .50:
        if temp > threshold:
            return True
        else:
            return False

def rollingAvg( lag, oldSet ):
    """
    Smooth list with lag value
    :param lag: int
    :param oldSet: list
    :return: list
    """

    newSet = []

    # insert lag-1 number of nans at beginning of list
    for i in range(0, lag - 1):
        newSet.append(Decimal('nan'))

    # calculate new values for list
    for i in range((lag - 1), len(oldSet)):
        sum = 0
        for j in range(lag):
            sum += oldSet[i - j]

        avg = sum / Decimal(lag)
        newSet.append(Decimal(avg))

    return newSet

def sortSet( set ):
    """
    Sort list of numbers in ascending order. Assumes that any nans are at the
    beginning of the list. Obviously doesn't sort starting nans
    :param set: list
    :return: list
    """

    # check if list starts with nan
    if np.isnan(float(set[0])):
        newSet = sorted(set[M_LAG-1:])
        set[M_LAG-1:] = newSet
    else:
        set = sorted(set)
    return set

def calcPercentile(percent, set):       #TESTED
    """
    Calculates percentile range (either above or below percentile) for set of
    temperatures. Returns the list of averaged mortality in the 4 days prior,
    all days of the heat wave, and 10 days after the end of the heat wave.
    :param percent: float/decimal
    :param set: list
    :return: list
    """

    # check for 100%
    if percent == Decimal('1.0'):
        return max(set)

    # convert percent to the appropriate index
    pValue = percent * len(set)

    set = sorted(set)

    # check for 0%
    if percent == Decimal('0'):
        return set[0]

    # check if percent is an integer
    if pValue % 1 == 0:

        # cast pValue as int so it can be used as an index
        pValue = int(pValue)

        # take average of values at indices percent and percent - 1
        return (set[pValue - 1] + set[pValue]) / Decimal('2')

    # if percentage needs to be rounded
    else:
        # round number up to nearest integer
        print "pValue: " + str(pValue)                                      # DELETE
        pValue = pValue.to_integral_exact(rounding=ROUND_CEILING)           # WHAT'S UP WITH THIS FUNCTION?
        print "rounded pValue: " + str(pValue)                              # DELETE
        pValue = int(pValue)

        return set[pValue - 1]

def findTempIndex( tempValue, dataSet ):
    dataSet = sorted(dataSet)
    index = 0
    while(dataSet[index] != tempValue):
        index += 1
    lowerIndex = index
    while(True):
        index += 1
        if( index<len(dataSet) and dataSet[index] != tempValue):
            break

def priorAnomalies( index, threshold, dataSet ):
    listLength = len(dataSet)

    if( index < 0 or index >= len( dataSet)):
        return 0

    for i in range(1,5):
        if(index-i > 0 and not(np.isnan(float( dataSet[i]))) and dataSet[index-i] >= threshold):
            return 0

    return 1

# test if a day is a leap day; TESTED
def isLeapDay(year, month, day):
    if (year == 1976 or year ==1980 or year == 1984 or year == 1988 or
        year == 1992 or year == 1996 or year == 2000 or year == 2004 or
        year == 2008 or year == 2012):
        if month == 2:
            if day == 29:
                return True

    return False

# find average for a specific day (e.g. avg mortality for Jan 1); TESTED
def findAvgOfDays(year, month, day, mortality, avgDayArr):
    del avgDayArr[:]
    for i in range(365):
        avgDayArr.append(Decimal('0'))

    leapDayAvg = 0
    dayCount = 0
    for i in range(len(mortality)):

        if(dayCount == 365):
            dayCount = 0
        if(isLeapDay(year[i],month[i],day[i])):
            #print "leap day found: " + str(year[i]) + " "+ str(month[i]) + " " + str(day[i]) + " " + str(mortality[i])
            leapDayAvg += mortality[i]
        else:
            avgDayArr[dayCount] += Decimal(mortality[i])
            dayCount += 1

    for i in range(365):
        avgDayArr[i] = avgDayArr[i]/Decimal(14)

    leapDayAvg = leapDayAvg/Decimal(4)
    return leapDayAvg

def findPercentileRange( percent, tempSet, mortalitySet, waveLen ):

    # check for invalid percentages
    if (percent <= 0 ):
        print "INVALID PERCENT! -- LESS THAN 0"
        return

    #initialize variables
    startingIndex = len(tempSet) - len(mortalitySet)
    totalWaveAvg = []
    tempWaveAvg = []
    rangeArr = []
    allWaveArr = []
    oneWaveArr = []
    preDays = [[],[],[],[]]
    postDays = [[],[],[],[],[],[],[],[],[],[]]
    numWaves = 0
    waveCheck = 0
    for i in range(waveLen):
        allWaveArr.append([])

    #calculate percentiles + temp list length
    listLen = len(tempSet)
    lowerVal = calcPercentile(percent, tempSet)

    #loop through temp array
    for i in range(startingIndex, listLen):
        if(     ( not np.isnan(float(tempSet[i])) )
            and ( not np.isnan(float(mortalitySet[i-startingIndex])))
            and tempInRange(tempSet[i], lowerVal, percent)
          ):
            waveCheck += 1
            tempWaveAvg.append(mortalitySet[i-startingIndex])
            oneWaveArr.append(mortalitySet[i-startingIndex])
        else:
            waveCheck = 0
            oneWaveArr = []
            tempWaveAvg = []

        if waveLen == waveCheck:
            numWaves += 1

            for j in range(waveLen):
                allWaveArr[j].append(oneWaveArr[j])

            #print "FOR HEAT WAVE AT INDECES " + str(i-waveLen + 1) + " to " + str(i)

            for j in range(10):
                if i + j + 1 < listLen:
                    postDays[j].append(Decimal(mortalitySet[i+j+1-startingIndex]))
                    #print "\t" + str(j+1) + " Day After: " + str(tempSet[i+j+1]) + "   .... | " + str(i+j+1)

            for j in range(4):
                if (i-j-startingIndex-waveLen >= 0) and not np.isnan(float(mortalitySet[i-j-startingIndex-waveLen])):
                    preDays[j].append(Decimal(mortalitySet[i-j-startingIndex-waveLen]))
                    #print "\t" + str(j+1) + " Day Before: " + str(tempSet[i-j-waveLen]) + "   .... | " + str(i-j-waveLen)

            totalWaveAvg.append( Decimal(sum(tempWaveAvg))/Decimal(len(tempWaveAvg)) )
            oneWaveArr = []
            tempWaveAvg = []
            waveCheck = 0

    print "NUMBER OF HEAT WAVES: " + str(numWaves)

    if numWaves == 0:
        for i in range(14+waveLen):
            rangeArr.append(Decimal('-10'))
        totalWaveAvg = -10
    else:
        #calculate preDays averages + add to rangeArr
        for i in range(4):
            preDays[3-i] = Decimal(sum(preDays[3-i]))/Decimal(len(preDays[3-i]))
            rangeArr.append(preDays[3-i])

        #calculate waveMortAvg + add to rangeArr
        for i in range(waveLen):
            allWaveArr[i] = Decimal(sum(allWaveArr[i]))/Decimal(len(allWaveArr[i]))
            rangeArr.append(allWaveArr[i])

        #calculate postDays averages + add to rangeArr
        for i in range(10):
            postDays[i] = Decimal(sum(postDays[i]))/Decimal(len(postDays[i]))
            rangeArr.append(postDays[i])

    #ST
        totalWaveAvg = Decimal(sum(totalWaveAvg))/Decimal(len(totalWaveAvg))
        allWaveArrAvg = Decimal(sum(allWaveArr))/Decimal(len(allWaveArr))

        #print "average is: " + str(totalWaveAvg) + "|" + str(allWaveArrAvg)
    #ET

    return totalWaveAvg, rangeArr


######################################################################

###"MAIN"
nyDict = pickle.load(open("shortCompiledNY.csv", 'rb'))

#initialize
subSmoothMortality = []
subMeanMortality = []
subDayMortality = []
dayAvgArr = []
M_LAG = 30

# setting up dicts and lists
mortality = nyDict['mortality']
minTemps = nyDict['minTemp']
maxTemps = nyDict['maxTemp']
meanTemps = nyDict['meanTemp']
year = nyDict['year']
month = nyDict['month']
day = nyDict['day']

# START RE-TESTING

# smoothing
smoothedMortality = rollingAvg(M_LAG, mortality)
smoothedMin = rollingAvg(3, minTemps)
smoothedMax = rollingAvg(3, maxTemps)
smoothedMean = rollingAvg(3, meanTemps)                                 #tested smooth are correct


#fill subSmoothMort (len 5085)
for index in range(M_LAG-1, len(smoothedMortality)):
    subSmoothMortality.append(mortality[index] - smoothedMortality[index]) #tested subSmooth are correct (just tested beginning and end)


print subSmoothMortality                            # DELETE

#fill subMeanMortality
meanMortality = sum(mortality)/Decimal(len(mortality))
#print "mean mortality is: " + str(meanMortality)                    #mean mortality is correct
for index in range(len(mortality)):
    subMeanMortality.append(mortality[index]-meanMortality)         #subMeanmortality is correct

#fill subDayMortality
testArr =[]
dayCount = 0
leapDayAvg = findAvgOfDays(year, month, day, mortality, dayAvgArr)      #test leapDayAvg + findAvgOfDays is correct
for i in range(len(mortality)):
    if(dayCount == 365):
        dayCount = 0
    if( not isLeapDay(year[i], month[i], day[i])):
        subDayMortality.append(mortality[i]-dayAvgArr[dayCount])
        dayCount += 1
    else:
        subDayMortality.append(mortality[i]-leapDayAvg)                 #subDayMortality correct

# END RE-TESTING

#ST
"""
smoothedTemp = rollingAvg(5, meanTemps)
print smoothedTemp
someLowerVal = calcPercentile(Decimal('0.8'), smoothedTemp[4:])
print "LOWER VAL: " + str(someLowerVal)

tempList = meanTemps
tempPercArr = [[],[],[],[],[]]
startPercent = Decimal('0.05')
for i in range(19):
    for j in range(5):
        tempPercAvg, fillerArr = findPercentileRange(startPercent, tempList, subSmoothMortality, j+1)
        tempPercArr[j].append(tempPercAvg)
        print "here"
    startPercent += Decimal('0.05')

xAxis = np.linspace(5,100,19)

# titles
plt.title("Average mortality anomaly during cold/hot event", fontsize = 15)
plt.ylabel("Average mortality anomaly", fontsize=15)
plt.xlabel("Percentile threshold", fontsize = 15)

# legend
red_patch = mpatches.Patch(color='red', label='1 day cold/hot event')
blue_patch = mpatches.Patch(color='blue', label='2 day cold/hot event')
green_patch = mpatches.Patch(color='green', label = '3 day cold/hot event')
black_patch = mpatches.Patch(color = 'black', label = '4 day cold/hot event')
magenta_patch = mpatches.Patch(color='magenta', label = '5 day cold/hot event')
plt.legend(handles=[red_patch, blue_patch, green_patch, black_patch, magenta_patch], loc = 'upper left')

# plot
plt.plot(xAxis, tempPercArr[0], color = 'red')
plt.plot(xAxis, tempPercArr[1], color = 'blue')
plt.plot(xAxis, tempPercArr[2], color = 'green' )
plt.plot(xAxis, tempPercArr[3], color = 'black')
plt.plot(xAxis, tempPercArr[4], color = 'magenta')
plt.show()
#"""

#"""
# graphing testing
#print "STANDARD DEVIATION OF SUBSMOOTHMORTALITY: " + str(np.std(subSmoothMort))
mortList = subSmoothMortality
waveAvg, range95 = findPercentileRange(Decimal('0.95'), meanTemps, mortList, 5)
waveAvg, range90 = findPercentileRange(Decimal('0.9'), meanTemps, mortList, 5)
waveAvg, range50 = findPercentileRange(Decimal('0.5'), meanTemps, mortList,5)
waveAvg, range10 = findPercentileRange(Decimal('0.1'), meanTemps, mortList, 5)
#waveAvg, rangeFour = findPercentileRange(Decimal('0.0'), Decimal('0.05'), meanTemps, mortList, 1)

fig = plt.figure(figsize=(10,7))
xAxis = np.linspace(-4,14,19)
plt.plot(xAxis, range95, color = "red")
#plt.plot(xAxis, range90, color = 'blue')
#plt.plot(xAxis, range50, color = "green")
#plt.plot(xAxis, range10, color = 'black')

# titles
plt.title("Mortality anomalies before, during, and after five \n consecutive days above the 95th temperature percentile", fontsize = 15)
plt.ylabel("Mortality anomaly", fontsize=15)
plt.xlabel("Days before, during, and after cold/hot event", fontsize = 15)

# legend
red_patch = mpatches.Patch(color='red', label='Above 95th percentile')
#blue_patch = mpatches.Patch(color='blue', label='Above 90th percentile')
#green_patch = mpatches.Patch(color='green', label = 'Above 50th percentile')
#black_patch = mpatches.Patch(color='black', label =  'Below 10th percentile')
#plt.legend(handles=[red_patch])

# plot
plt.show()
#"""

'''
#SHOW GRAPHS OF ALL THREE ALTERED MORTALITY SETS

xAxis = np.linspace(1987, 2000, 5114)
#plt.plot(xAxis, subMeanMortality)
plt.plot(xAxis, subDayMortality)
xAxis = np.linspace(1987, 2000, 5085)
#plt.plot(xAxis, subSmoothMort)
plt.show()
'''