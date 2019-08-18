import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize as opt
import csv
from decimal import Decimal
import decimal
from scipy import stats
from sklearn import linear_model
import math
import matplotlib.patches as mpatches

def sameSeason( pMonth, cMonth ):
    """
    Check if two nums representing months are within the same season
    :param pMonth: int
    :param cMonth: int
    :return: bool
    """
    if pMonth == 12 or pMonth == 1 or pMonth == 2:
        if cMonth == 12 or cMonth == 1 or cMonth == 2:
            return True
        else:
            return False

    if pMonth == 3 or pMonth == 4 or pMonth == 5:
        if cMonth == 3 or cMonth == 4 or cMonth == 5:
            return True
        else:
            return False

    if pMonth == 6 or pMonth == 7 or pMonth == 8:
        if cMonth == 6 or cMonth == 7 or cMonth == 8:
            return True
        else:
            return False

    if pMonth == 9 or pMonth == 10 or pMonth == 11:
        if cMonth == 9 or cMonth == 10 or cMonth == 11:
            return True
        else:
            return False

"""
def calcPercentile(percent, set):       #TESTED

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
        print "|" + str(pValue)                                                        # DELETE
        pValue = pValue.to_integral_exact(rounding=ROUND_CEILING)           # WHAT'S UP WITH THIS FUNCTION?
        print pValue                                                        # DELETE
        pValue = int(pValue)

        return set[pValue - 1]
"""

def splitMortYearly(year, mortality):
    yearlyMort = []
    tempMortList = []

    index = 0
    previousYr = year[index]

    while index < len(year):
        currentYr = year[index]

        if currentYr != previousYr:
            yearlyMort.append(tempMortList)
            tempMortList = []
            previousYr = currentYr

        tempMortList.append(mortality[index])
        index += 1

    yearlyMort.append(tempMortList)
    return yearlyMort

def splitMortYearlyNan(year, mortality):
    yearlyMort = []
    tempMortList = []

    index = 0
    previousYr = year[index]

    while index < len(year):
        currentYr = year[index]

        if currentYr != previousYr:
            if currentYr - previousYr > 1:
                for i in range(currentYr-previousYr-1):
                    yearlyMort.append(['nan'])

            yearlyMort.append(tempMortList)
            tempMortList = []
            previousYr = currentYr

        tempMortList.append(mortality[index])
        index += 1

    yearlyMort.append(tempMortList)
    return yearlyMort

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

def calcSeasonModified( monthNum ):
    """
    Calculate season "index" (DJF = 0, JJA = 1, MAM and SON = 3) but only for
    winter and summer.
    :param monthNum: int
    :return: int
    """

    if monthNum == 12 or monthNum == 1 or monthNum == 2:
        return 0

    elif monthNum == 6 or monthNum == 7 or monthNum == 7:
        return 1

    else:
        return 3

def celsiusToFahrenheit( oldList ):
    for i in range(len(oldList)):
        oldList[i] = (np.float64(oldList[i] * 1.8))+ 32
    return oldList

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
        return (set[pValue - 1] + set[pValue])/Decimal('2')

    # if percentage needs to be rounded
    else:
        # round number up to nearest integer
        print pValue                                                        # DELETE
        pValue = pValue.to_integral_exact(rounding=decimal.ROUND_CEILING)           # WHAT'S UP WITH THIS FUNCTION?
        print pValue                                                        # DELETE
        pValue = int(pValue)

        return set[pValue - 1]

def makeYearlySeasonalList(startIndex, endIndex, numWinterYears, month, origList):
    newList = [[],[]]
    for i in range(2):
        for j in range(numWinterYears):
            newList[i].append([])
    newList[1].append([])

    tempList = []
    yearIndex = 0
    index = startIndex
    currentSeason = currentMonth = 0

    while index <= endIndex:
        currentSeason = month[index]
        currentMonth = month[index]

        # iterate through a season
        while (sameSeason(currentSeason, currentMonth)) and index < len(month):
            currentMonth = month[index]

            # add to temp lists
            tempList.append(origList[index])

            # update index and previousMonth
            index += 1
            if index < len(month):
                currentMonth = month[index]

        seasonIndex = calcSeasonModified(currentSeason)
        if seasonIndex < 3:
            newList[seasonIndex][yearIndex] = tempList
            if seasonIndex == 0:
                yearIndex += 1

        # clear temp lists
        tempList = []

    return newList

def isHeatWave(startIndex, tempList, waveLen, tempThreshold): # DO NOT USE SOMETHING IS WRONG
    for i in range(1 + waveLen):
        if startIndex + i > len(tempList):
            return False

        if tempList[startIndex + i] < tempThreshold:
            return False

    return True


# MAIN

# read in mortality and temperature data
nyDict = pickle.load(open("shortCompiledNY.csv", 'rb'))

# setting up dicts and lists
mortality = nyDict['mortality']
minTemps = nyDict['minTemp']
maxTemps = nyDict['maxTemp']
meanTemps = nyDict['meanTemp']
year = nyDict['year']
month = nyDict['month']
day = nyDict['day']
weekday = nyDict['weekday']
meanDewPts = nyDict['meanDewPt']

# read in hist data
histDict = pickle.load(open("gfdlHistCompiled.csv", 'rb'))
histTemp = histDict['meanTemps']
histDewPt = histDict['dewPts']
histYear = histDict['year']
histTemp = celsiusToFahrenheit(histTemp)
histDewPt =[i+np.float64(273.15) for i in histDewPt]
histDewPt = celsiusToFahrenheit(histDewPt)
hSmoothMeanTemp5 = rollingAvg(5, [Decimal(i) for i in histTemp])
hSmoothMeanTemp5 = [np.float64(i) for i in hSmoothMeanTemp5]

# initialize
smoothMort = []
subSmoothMort = []
smoothMeanTemp5 = []
smoothMeanTemp4 = []
smoothMeanTemp3 = []
smoothMaxTemp5 = []
smoothMinTemp5 = []
annualAvgMort = []

# smooth temperature set
smoothMort = rollingAvg(30, mortality)
smoothMeanTemp5 = rollingAvg(5, meanTemps)                               # change this as desired
smoothMeanTemp4 = rollingAvg(4, meanTemps)
smoothMeanTemp3 = rollingAvg(3, meanTemps)
smoothMinTemp5 = rollingAvg(5, minTemps)
smoothMaxTemp5 = rollingAvg(5, maxTemps)

# create subSmoothMort list
for i in range(len(smoothMort)):
    if smoothMort[i] == Decimal('nan'):
        subSmoothMort.append(Decimal('nan'))
    else:
        subSmoothMort.append(Decimal(mortality[i] - smoothMort[i]))

percent = Decimal('0.95')

sLowerMeanTemp = calcPercentile(percent, smoothMeanTemp5[4:])
sLowerDewPt = calcPercentile(percent, meanDewPts)

hLowerMeanTemp = calcPercentile(percent, hSmoothMeanTemp5[4:])
#hLowerDewPt = calcPercentile(Decimal('0.9'), histDewPt)

# cast temp and mortality lists as floats
for i in range(len(smoothMort)):
    smoothMeanTemp5[i] = np.float64(smoothMeanTemp5[i])
    smoothMeanTemp4[i] = np.float64(smoothMeanTemp4[i])
    smoothMeanTemp3[i] = np.float64(smoothMeanTemp3[i])
    smoothMinTemp5[i] = np.float64(smoothMinTemp5[i])
    smoothMaxTemp5[i] = np.float64(smoothMaxTemp5[i])
    subSmoothMort[i] = np.float64(subSmoothMort[i])
    meanTemps[i] = np.float64(meanTemps[i])
    minTemps[i] = np.float64(minTemps[i])
    maxTemps[i] = np.float64(maxTemps[i])
    smoothMort[i] = np.float64(smoothMort[i])          # DELETE

    # mort, smoothMort, subSmoothMort all tested as correct (excel sheet)

# create annualAvgMort list
i = 0
currentYr = year[0]
yrStart = yrEnd = 0
while i < len(year):
    loopYr = year[i]

    if loopYr == currentYr:
        yrEnd = i
        i += 1
    else:
        annualAvgMort.append(sum(mortality[yrStart:yrEnd + 1])/np.float64(len(mortality[yrStart:yrEnd+1])))
        yrStart = i
        i += 1
        currentYr = year[i]

# add last year of mortality
annualAvgMort.append(sum(mortality[yrStart:yrEnd + 1])/np.float64(len(mortality[yrStart:yrEnd + 1])))
    # annual AvgMort is correct

startIndex = 0
endIndex = 0

# find first season of first year and last season of last year
for i in range(len(day)):
    if year[i] == 1987 and (month[i] < 6):
        startIndex = i + 1
    if year[i] == 2000 and month[i] == 12:
        endIndex = i - 1
        break
    # start and end indeces are correct (151 and 5082)

"""
    this part is for winter/summer by year (below) for the entire time span
"""
numWinterYears = 2000-1987

#initialize
dailyMins5 = makeYearlySeasonalList(startIndex, endIndex, numWinterYears, month, smoothMinTemp5)                    # 0th index is winter
dailyMeans5 = makeYearlySeasonalList(startIndex, endIndex, numWinterYears, month, smoothMeanTemp5)
dailyMeans4 = makeYearlySeasonalList(startIndex, endIndex, numWinterYears, month, smoothMeanTemp4)
dailyMeans3 = makeYearlySeasonalList(startIndex, endIndex, numWinterYears, month, smoothMeanTemp3)
dailyMaxs5 = makeYearlySeasonalList(startIndex, endIndex, numWinterYears, month, smoothMaxTemp5)

dailyUnsmthMins = makeYearlySeasonalList(startIndex, endIndex, numWinterYears, month, minTemps)
dailyUnsmthMeans = makeYearlySeasonalList(startIndex, endIndex, numWinterYears, month, meanTemps)
dailyUnsmthMaxs = makeYearlySeasonalList(startIndex, endIndex, numWinterYears, month, maxTemps)

dailySubSmoothMort = makeYearlySeasonalList(startIndex, endIndex, numWinterYears, month, subSmoothMort)
dailyUnsmthMort = makeYearlySeasonalList(startIndex, endIndex, numWinterYears, month, mortality)

dailyWeekday = makeYearlySeasonalList(startIndex, endIndex, numWinterYears, month, weekday)

dailyDewPts = makeYearlySeasonalList(startIndex, endIndex, numWinterYears, month, meanDewPts)



"""
    [5]
    Using days that only exceed 90th/95th (did both - but stick with top 5%) percentile
    overlapping graphs
"""
#"""
histPredictMort = []
tempPercList = []
dewPercList = []
mortPerclist = []
weekdayPercList = []
histTempPercList = []
histDewPercList = []

histTempList = hSmoothMeanTemp5[4:]
histDewPt = histDewPt[4:]
mortList = subSmoothMort[29:]
tempList = smoothMeanTemp5[29:]
dewPtList = meanDewPts[29:]
weekdayList = weekday[29:]

waveLen = 1
count = 0

# find all days in top 5% for historical data
for i in range(len(histTempList)):
    if histTempList[i] > hLowerMeanTemp and isHeatWave(i, histTempList, waveLen, hLowerMeanTemp):
        histTempPercList.append(histTempList[i])
        histDewPercList.append(histDewPt[i])
        count += 1
print count

# find all days in top 5%
for i in range(len(tempList)):
    if tempList[i] > sLowerMeanTemp:
        tempPercList.append(tempList[i])
        dewPercList.append(dewPtList[i])
        mortPerclist.append(mortList[i])
        weekdayPercList.append(weekdayList[i])

numDays = len(tempPercList)

numEightyPerc = int(math.ceil(0.8 * numDays))  #204
numTwentyPerc = int(0.2 * numDays)  # 234
numMeasures = 2

# list len all checked by hand

# initialize
pActualMort = []
predictMort = []
trainMeasures = []
testMeasures = []
histMeasures = []
coeffs = []
intercepts = []
error = []
fStart = fEnd = listCount = pStart = pEnd = 0

rTotal = 0
for i in range(numMeasures):
    trainMeasures.append([])
    histMeasures.append([])
    testMeasures.append([])
    coeffs.append([])

histMeasures[0] = histTempPercList
histMeasures[1] = histDewPercList

for i in range(numTwentyPerc + 1):
    regr = linear_model.LinearRegression()

    # fit
    fStart = i
    fEnd = numDays - 1 - numTwentyPerc + i

    trainMeasures[0] = tempPercList[fStart:fEnd+1]
    trainMeasures[1] = dewPercList[fStart:fEnd+1]

    # fit
    regr.fit((np.transpose(trainMeasures)).reshape(numEightyPerc, numMeasures), (np.transpose(mortPerclist[fStart:fEnd + 1])).reshape(numEightyPerc, 1))

    # gather regr coefficients and intercepts
    for j in range(numMeasures):
        coeffs[j].append(regr.coef_[0][j])
    print regr.intercept_
    intercepts.append(regr.intercept_[0])

    pStart = fEnd + 1
    pEnd = numDays - 1

    #print "interval",
    #print pStart, pEnd,
    #print "   || ",

    # add temp and weekday data to testMeasures
    testMeasures[0] = tempPercList[pStart:pEnd+1]
    testMeasures[1] = dewPercList[pStart:pEnd+1]

    # fill actual mortality to compare with predicted values
    pActualMort = mortPerclist[pStart:pEnd + 1]

    # reset pStart and pEnd values
    pStart = 0
    pEnd = i - 1


    print pStart, pEnd,

    # add temp and weekday data to testMeasures
    testMeasures[0] = list(testMeasures[0] + tempPercList[pStart:pEnd+1])
    testMeasures[1] = list(testMeasures[1] + dewPercList[pStart:pEnd+1])

    # fill actual mortality to compare with predicted values
    pActualMort = list(pActualMort + mortPerclist[pStart:pEnd + 1])

    # predict values
    predictMort = regr.predict((np.transpose(testMeasures)).reshape(numTwentyPerc, numMeasures))
    predictMort = (np.transpose(predictMort)[0].tolist())

    # predict values historical data
    histPredictMort = regr.predict((np.transpose(histMeasures)).reshape(len(histMeasures[0]), numMeasures))
    histPredictMort = (np.transpose(histPredictMort)[0].tolist())

    #plt.scatter(histMeasures[0], histPredictMort, color='green')
    #plt.scatter(testMeasures[0], pActualMort, color = 'blue')

    #calulating RMSE
    rms = 0
    #print "df is: " + str(len(testMeasures[0]))`
    for i in range(len(testMeasures)):
        rms += ((predictMort[i] - pActualMort[i]) ** 2)
    #print "RMS IS: " + str(np.sqrt(rms / len(predictMort)))
    rVal = regr.score((np.transpose(testMeasures)).reshape(numTwentyPerc, numMeasures), pActualMort)
    rVal = np.sqrt(np.absolute(rVal))
    print "R VALUE IS: " + str(rVal)

    plt.scatter(testMeasures[0], predictMort, color = 'green')
    plt.scatter(testMeasures[0], pActualMort, color = 'blue')
    #plt.show()

    rTotal = rTotal + rVal


    #plt.scatter(testMeasures[0], predictMort, color='red')
    #plt.show()

#print rvalue average
print "r^2 avg is: " + str(rTotal/(numTwentyPerc))
print "total data points are: " + str(numEightyPerc+numTwentyPerc)

plt.xlabel("Temperature ($^\circ$F)", fontsize = 15)
plt.ylabel("Daily mortality anomaly", fontsize = 15)
plt.title("Model trained with portion of observational data (superimposed)", fontsize =15)
blue_patch = mpatches.Patch(color='blue', label='Observational data')
green_patch = mpatches.Patch(color='green', label = 'Model projections')
plt.legend(handles=[blue_patch, green_patch], loc='upper left')
#plt.show()

for i in range(numMeasures):
    coeffs[i] = np.float64(sum(coeffs[i])) / len(coeffs[i])

intercepts = np.float64(sum(intercepts))/len(intercepts)

print coeffs
print intercepts

print "HERE"
print len(tempPercList)
