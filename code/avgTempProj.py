import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize as opt
import csv
from decimal import Decimal
import decimal
from scipy import stats
from sklearn import linear_model
import os
import math


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

# initialize
summerHistTemp = []
winterHistTemp = []

summerPastTemp = []
winterPastTemp = []

tempList = []
modelCount = 0

# READ IN FUTURE PROJECTIONS
currentDirPath = os.getcwd()
subDir = os.path.join(currentDirPath, "allFutModelTemps\\")
subDir = os.listdir(subDir)

for i in range(len(subDir)):
    summerHistTemp.append([])
    winterHistTemp.append([])

for file in subDir:
    filePath = "C:/Users/h/PycharmProjects/untitled2/allFutModelTemps/" + file
    histDict = pickle.load(open(filePath, 'rb'))

    # import data
    tempYearList = histDict['year']
    tempMonthList = histDict['month']
    tempList = histDict['meanTemps']
    tempList = celsiusToFahrenheit(tempList)

    # start and end indices for 2020-2080
    endIndex = startIndex = 0
    for i in range(len(tempMonthList)):
        if tempYearList[i] == 2020 and (tempMonthList[i] < 6):
            startIndex = i + 1
        if tempYearList[i] == 2080 and tempMonthList[i] == 12:
            endIndex = i - 1
            break
    numWinterYears = 2080 - 2020

    # split into yearly lists
    tempList = makeYearlySeasonalList(startIndex, endIndex, numWinterYears, tempMonthList, tempList)

    summerHistTemp[modelCount] = tempList[1]
    winterHistTemp[modelCount] = tempList[0]

    tempList = []
    modelCount += 1

# READ IN PAST PROJECTIONS
modelCount = 0
currentDirPath = os.getcwd()
subDir = os.path.join(currentDirPath, "allHistModelTemps\\")
subDir = os.listdir(subDir)

for i in range(len(subDir)):
    summerPastTemp.append([])
    winterPastTemp.append([])

for file in subDir:
    filePath = "C:/Users/h/PycharmProjects/untitled2/allHistModelTemps/" + file
    pastDict = pickle.load(open(filePath, 'rb'))

    # import data
    tempYearList = pastDict['year']
    tempMonthList = pastDict['month']
    tempList = pastDict['meanTemps']
    tempList = celsiusToFahrenheit(tempList)

    # start and end indices for 2020-2080
    endIndex = startIndex = 0
    for i in range(len(tempMonthList)):
        if tempYearList[i] == 1987 and (tempMonthList[i] < 6):
            startIndex = i + 1
        if tempYearList[i] == 2000 and tempMonthList[i] == 12:
            endIndex = i - 1
            break
    numWinterYears = 2000 - 1987

    # split into yearly lists
    tempList = makeYearlySeasonalList(startIndex, endIndex, numWinterYears, tempMonthList, tempList)

    summerPastTemp[modelCount] = tempList[1]
    winterPastTemp[modelCount] = tempList[0]

    tempList = []
    modelCount += 1

for i in range(len(summerHistTemp)):
    for j in range(len(summerHistTemp[i])):
        if len(summerHistTemp[i][j]) != 92:
            summerHistTemp[i][j].insert(0,'nan')
            summerHistTemp[i][j].insert(0, 'nan')

for i in range(len(winterHistTemp)):
    for j in range(len(winterHistTemp[i])):
        if len(winterHistTemp[i][j]) < 90:
            numMissing = 90 - len(winterHistTemp[i][j])
            for k in range(numMissing):
                winterHistTemp[i][j].insert(0,'nan')

        elif len(winterHistTemp[i][j]) > 90:
            numExtra = len(winterHistTemp[i][j]) - 90
            for k in range(numExtra):
                del winterHistTemp[i][j][0]

#correct up to here

# FOR SUMMMER EXTREMES
histBoxPlot = []
pastBoxPlot = []

histYrMax = []
pastYrMax = []


# for every year
for i in range(len(winterHistTemp[0])):
    # for every model
    for j in range(len(winterHistTemp)):
        histYrMax.append(np.float64(sum(winterHistTemp[j][i]))/np.float64(len(winterHistTemp[j][i])))   # list of all mortality proj from the models
    histYrMax.sort()
    histBoxPlot.append(histYrMax)
    histYrMax = []

for i in range(len(winterPastTemp[0])):
    # for every model
    for j in range(len(winterPastTemp)):
        pastYrMax.append(np.float64(sum(winterPastTemp[j][i]))/np.float64(len(winterPastTemp[j][i])))     # list of all mortality proj from the models
    pastYrMax.sort()
    pastBoxPlot.append(pastYrMax)
    pastYrMax = []


# appended together into list approach
fillerList = []
for i in range(20):
    fillerList.append(np.float64(0))
pastFutList = pastBoxPlot + fillerList + histBoxPlot
print len(pastFutList)
xScale = []
for i in range(1988,2081):
    print i
    if i%4 == 0:
        xScale.append(str(i))
    else:
        xScale.append("")

fig = plt.figure(figsize=(15,8))
boxPlot = plt.boxplot(pastFutList, whis =[10,90])
futTicks = plt.xticks(np.arange(1,94), xScale, rotation=45)
plt.ylim(30,60)
plt.title("Average winter temperatures across CMIP5 model projections", fontsize = 15)
plt.ylabel("Temperature ($^\circ$F)", fontsize = 15)
plt.xlabel("Year", fontsize = 15)
plt.show()

#FOR SUMMER AVG




