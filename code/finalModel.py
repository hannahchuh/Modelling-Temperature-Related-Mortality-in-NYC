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

def isWithinThreshold(percent, testValue, threshold):
    if percent < Decimal('0.5'):
        print "less than"
        if testValue < threshold:
            return True
        else:
            return False
    if percent > Decimal('0.5'):
        print "more than"
        if testValue > threshold:
            return True
        else:
            return False


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

def splitMortYearlyNan(year, mortality, season):
    yearlyMort = []
    tempMortList = []

    index = 0
    previousYr = year[index]

    numPrecedingNan = 0
    if year[index] > 2010:
        if (season == 0 and year[index] != 2021):
            numPrecedingNan = year[index] - 2021
        if (season == 1 and year[index] != 2020):
            numPrecedingNan = year[index] - 2020
    else:
        if (season == 0 and year[index] != 1988):
            numPrecedingNan = year[index] - 1988
        if (season == 1 and year[index] != 1987):
            numPrecedingNan = year[index] - 1987

    for i in range(numPrecedingNan):
        yearlyMort.append(['nan'])

    while index < len(year):
        currentYr = year[index]

        if currentYr != previousYr:
            if currentYr - previousYr > 1:
                for i in range(currentYr - previousYr - 1):
                    yearlyMort.append(['nan'])

            yearlyMort.append(tempMortList)
            tempMortList = []
            previousYr = currentYr

        tempMortList.append(mortality[index])
        index += 1

    yearlyMort.append(tempMortList)

    numAfterNan = 0
    if year[index - 1 ] > 2010 and year[index - 1] != 2080:
        numAfterNan = 2080 - year[index - 1]
    elif year[index-1] != 2000:
        numAfterNan = 2000 - year[index - 1]

    for i in range(numAfterNan):
        yearlyMort.append(['nan'])

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
        return np.float64(set[pValue - 1] + set[pValue])/np.float64('2')

    # if percentage needs to be rounded
    else:
        # round number up to nearest integer
        #print pValue                                                        # DELETE
        pValue = pValue.to_integral_exact(rounding=decimal.ROUND_CEILING)           # WHAT'S UP WITH THIS FUNCTION?
        #print pValue                                                        # DELETE
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
modelName = "cnrm"
fileName = modelName + "FutureCompiled.csv"
histDict = pickle.load(open(fileName, 'rb'))
histTemp = histDict['meanTemps']
histDewPt = histDict['dewPts']
histMonth = histDict['month']
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

# smooth temperature set
smoothMort = rollingAvg(30, mortality)
smoothMeanTemp5 = rollingAvg(5, meanTemps)                               # change this as desired

# create subSmoothMort list
for i in range(len(smoothMort)):
    if smoothMort[i] == Decimal('nan'):
        subSmoothMort.append(Decimal('nan'))
    else:
        subSmoothMort.append(Decimal(mortality[i] - smoothMort[i]))

percent = Decimal('0.90')

#sLowerMeanTemp = calcPercentile(percent, smoothMeanTemp5[4:])
#hLowerMeanTemp = calcPercentile(percent, hSmoothMeanTemp5[4:])

# cast temp and mortality lists as floats
for i in range(len(smoothMort)):
    smoothMeanTemp5[i] = np.float64(smoothMeanTemp5[i])
    subSmoothMort[i] = np.float64(subSmoothMort[i])
    meanTemps[i] = np.float64(meanTemps[i])
    minTemps[i] = np.float64(minTemps[i])
    maxTemps[i] = np.float64(maxTemps[i])
    smoothMort[i] = np.float64(smoothMort[i])          # DELETE

    # mort, smoothMort, subSmoothMort all tested as correct (excel sheet)

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

numWinterYears = 2000-1987

#initialize seaononal lists                 # 0th index is winter
dailyMeans5 = makeYearlySeasonalList(startIndex, endIndex, numWinterYears, month, smoothMeanTemp5)
dailySubSmoothMort = makeYearlySeasonalList(startIndex, endIndex, numWinterYears, month, subSmoothMort)
dailyDewPts = makeYearlySeasonalList(startIndex, endIndex, numWinterYears, month, meanDewPts)

IS_FUTURE_SET = True
if len(histMonth) <= 5114:
    IS_FUTURE_SET = False

if IS_FUTURE_SET:
    # start and end indices for 2020-2080
    for i in range(len(histMonth)):
        if histYear[i] == 2020 and (histMonth[i] < 6):
            startIndex = i + 1
        if histYear[i] == 2080 and histMonth[i] == 12:
            endIndex = i - 1
            break
    numWinterYears = 2080 - 2020
else:
    # start and end indices for 1987-2000
    endIndex = endIndex - 4                                                                                     #to account for missing leap years - 4
print startIndex
print endIndex
dailyHistMeans5 = makeYearlySeasonalList(startIndex, endIndex, numWinterYears, histMonth, hSmoothMeanTemp5)
dailyHistDewPts = makeYearlySeasonalList(startIndex, endIndex, numWinterYears, histMonth, histDewPt)


#make year perc list
dailyYear = [[], []]
dailyHistYear = [[], []]

startingYear = 1988
for i in range(len(dailyMeans5)):
    loopingYear = startingYear
    for j in range(len(dailyMeans5[i])):
        dailyYear[i].append([])
        for k in range(len(dailyMeans5[i][j])):
            dailyYear[i][j].append(loopingYear)
        loopingYear += 1
    startingYear -= 1

if IS_FUTURE_SET:                       # starting off at winter
    startingYear = 2021
else:
    startingYear = 1988

for i in range(len(dailyHistMeans5)):
    loopingYear = startingYear
    for j in range(len(dailyHistMeans5[i])):
        dailyHistYear[i].append([])
        for k in range(len(dailyHistMeans5[i][j])):
            dailyHistYear[i][j].append(loopingYear)
        loopingYear += 1
    startingYear -= 1


#make yearly percentiles
yearlyPercentiles = [[],[]]
hYearlyPercentiles = [[],[]]

for i in range(len(dailyMeans5)):
    for j in range(len(dailyMeans5[i])):
        yearlyPercentiles[i].append(calcPercentile(percent, dailyMeans5[i][j]))

for i in range(len(dailyHistMeans5)):
    for j in range(len(dailyHistMeans5[i])):
        hYearlyPercentiles[i].append(calcPercentile(percent, dailyHistMeans5[i][j]))

# specify winter or summer
season = 1

# initialize
tempPercList = []
mortPercList = []
dewPercList = []
yearPercList = []

histPredictMort = []
histTempPercList = []
histDewPercList = []
histYearPercList = []

tempCopy = []
dewCopy = []
yearCopy = []
mortCopy = []

"""
# for threshold based on season

# for every hist year
for i in range(len(dailyHistMeans5[season])):
    # for every day in the year
    for j in range(len(dailyHistMeans5[season][i])):
        if isWithinThreshold(percent, dailyHistMeans5[season][i][j], hYearlyPercentiles[season][i]):
        #if dailyHistMeans5[season][i][j] < hYearlyPercentiles[season][i]:
            tempCopy.append(dailyHistMeans5[season][i][j])
            dewCopy.append(dailyHistDewPts[season][i][j])
            yearCopy.append(dailyHistYear[season][i][j])
    dailyHistMeans5[season][i] = tempCopy
    dailyHistDewPts[season][i] = dewCopy
    dailyHistYear[season][i] = yearCopy
    tempCopy = []
    dewCopy = []
    yearCopy = []

# for every year
for i in range(len(dailyMeans5[season])):
    #for every day in the year
    for j in range(len(dailyMeans5[season][i])):
        if isWithinThreshold(percent, dailyMeans5[season][i][j], yearlyPercentiles[season][i]):
            print dailyMeans5[season][i][j],
            print yearlyPercentiles[season][i]
            tempCopy.append(dailyMeans5[season][i][j])
            dewCopy.append(dailyDewPts[season][i][j])
            yearCopy.append(dailyYear[season][i][j])
            mortCopy.append(dailySubSmoothMort[season][i][j])
    dailyMeans5[season][i] = tempCopy
    dailyDewPts[season][i] = dewCopy
    dailyYear[season][i] = yearCopy
    dailySubSmoothMort[season][i] = mortCopy
    tempCopy = []
    dewCopy = []
    yearCopy = []
    mortCopy = []
"""

#append all seasonal lists into the perc list (either seasonal threshold or no threshold)
for i in range(len(dailyMeans5[season])):
    tempPercList = list(tempPercList + dailyMeans5[season][i])
    dewPercList= list(dewPercList + dailyDewPts[season][i])
    mortPercList = list(mortPercList + dailySubSmoothMort[season][i])
    yearPercList = list(yearPercList + dailyYear[season][i])

for i in range(len(dailyHistMeans5[season])):
    histTempPercList= list(histTempPercList+ dailyHistMeans5[season][i])
    histDewPercList= list(histDewPercList+ dailyHistDewPts[season][i])
    histYearPercList = list(histYearPercList + dailyHistYear[season][i])

#"""
# for threshold based on lowerVal and hLowerVal
lowerVal = calcPercentile(percent, tempPercList)                                    # QUESTION: THis is based on all SUMMER or WINTER seasons - make it based on ALL seasson?
hLowerVal = lowerVal
#hLowerVal = calcPercentile(percent, histTempPercList)                               # if yes then uncomment below defining percent (line 299)

# historical percent list
for i in range(len(histTempPercList)):
    if isWithinThreshold(percent, histTempPercList[i], hLowerVal):
    #if dailyHistMeans5[season][i][j] < hYearlyPercentiles[season][i]:
        tempCopy.append(histTempPercList[i])
        dewCopy.append(histDewPercList[i])
        yearCopy.append(histYearPercList[i])
histTempPercList = tempCopy
histDewPercList = dewCopy
histYearPercList = yearCopy
tempCopy = []
dewCopy = []
yearCopy = []

# percent list
for i in range(len(tempPercList)):
    if isWithinThreshold(percent, tempPercList[i], lowerVal):
        tempCopy.append(tempPercList[i])
        dewCopy.append(dewPercList[i])
        yearCopy.append(yearPercList[i])
        mortCopy.append(mortPercList[i])
tempPercList = tempCopy
dewPercList = dewCopy
yearPercList = yearCopy
mortPercList = mortCopy

#"""

# initialize for model
pActualMort = []
predictMort = []
mortTrain = []
trainMeasures = []
testMeasures = []
histMeasures = []
coeffs = []
intercepts = []
error = []
fStart = fEnd = listCount = pStart = pEnd = 0

numMeasures = 2
numDays = len(tempPercList)

for i in range(numMeasures):
    trainMeasures.append([])
    histMeasures.append([])
    testMeasures.append([])
    coeffs.append([])

histMeasures[0] = histTempPercList
histMeasures[1] = histDewPercList

regr = linear_model.LinearRegression()

trainMeasures[0] = tempPercList
trainMeasures[1] = dewPercList
mortTrain = mortPercList

# fit
regr.fit((np.transpose(trainMeasures)).reshape(numDays, numMeasures), (np.transpose(mortTrain)).reshape(numDays, 1))

# gather regr coefficients and intercepts
for j in range(numMeasures):
    coeffs[j].append(regr.coef_[0][j])
intercepts = regr.intercept_[0]


# add temp and weekday data to testMeasures
testMeasures[0] = tempPercList
testMeasures[1] = dewPercList

# fill actual mortality to compare with predicted values
pActualMort = mortPercList

# predict values
predictMort = regr.predict((np.transpose(trainMeasures)).reshape(numDays, numMeasures))
predictMort = (np.transpose(predictMort)[0].tolist())

# predict values historical data
histPredictMort = regr.predict((np.transpose(histMeasures)).reshape(len(histMeasures[0]), numMeasures))
histPredictMort = (np.transpose(histPredictMort)[0].tolist())

#plt.scatter(testMeasures[0], pActualMort)
plt.scatter(testMeasures[0], predictMort, color='red')
plt.scatter(histMeasures[0], histPredictMort, color='green')
plt.show()

print coeffs
print intercepts

#looking at mortality per year
hYearlyMort = splitMortYearlyNan(histYearPercList, histPredictMort, season)
pYearlyMort = splitMortYearlyNan(yearPercList, predictMort, season)
print len(pYearlyMort)

for i in range(len(hYearlyMort)):
    if hYearlyMort[i] == ['nan']:
        hYearlyMort[i] = 'nan'
    else:
        hYearlyMort[i] = sum(hYearlyMort[i])  # don't divide by len if you want to see sums

for i in range(len(pYearlyMort)):
    if pYearlyMort[i] == ['nan']:
        pYearlyMort[i] = 'nan'
    else:
        pYearlyMort[i] = sum(pYearlyMort[i])

plt.hold(True)
if season == 0:
    startingYear = 1988
else:
    startingYear = 1987
xAxis = np.arange(startingYear,2001)
plt.scatter(xAxis, pYearlyMort, color = 'red')
if len(hYearlyMort) > 14:
    if season == 0:
        xAxis = np.arange(2021, 2081)
    else:
        xAxis = np.arange(2020,2081)
plt.scatter(xAxis, hYearlyMort, color = 'blue')
plt.show()

# pickle future mortality proj
futureDict = {}
fileName = modelName + "FutureModelMort"
if season == 0:
    fileName += "Winter.csv"
else:
    fileName += "Summer.csv"
futureDict.update({'mortality':hYearlyMort})
with open(fileName, 'wb') as handle:
    pickle.dump(futureDict, handle)

# write future mortality proj to readable csv file
sampleDict = futureDict
dictLength = len(sampleDict)
tempList = []
fileName = modelName + "FutureModelMortReadable"
if season == 0:
    fileName += "Winter.csv"
else:
    fileName += "Summer.csv"
with open(fileName, "wb") as fileObj:
    fileWriter = csv.writer(fileObj)
    listLength = len(sampleDict.itervalues().next())
    for index in range(listLength):
        for key in sampleDict:
            tempList.append( sampleDict[key][index])
        fileWriter.writerow(tempList)
        tempList = []