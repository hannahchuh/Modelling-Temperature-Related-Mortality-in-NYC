"""
    <description>
"""

# LIBRARIES
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize as opt
import csv
from decimal import Decimal
from scipy import stats
from sklearn import linear_model
import math


# FUNCTIONS
def firstOrder(x, a, b):
    return a + b*x

def secondOrder(x, a, b, c):
    return a + b*x + c*x*x

def thirdOrder(x, a, b, c, d):
    return a + b*x + c*x*x + d*x*x*x

def fourthOrder(x, a, b, c, d, e):
    return a + b*x + c*x*x + d*x*x*x + e*x*x*x*x

def fifthOrder(x, a, b, c, d, e, f):
    return a + b*x + c*x*x + d*x*x*x + e*x*x*x*x + f*x*x*x*x*x

def sixthOrder(x, a, b, c, d, e, f, g):
    return a + b*x + c*(x**2) + d*(x**3) + e*(x**4) + f*(x**5) + g*(x**6)

def seventhOrder(x, a, b, c, d, e, f, g, h):
    return a + b*x + c*(x**2) + d*(x**3) + e*(x**4) + f*(x**5) + g*(x**6) + h*(x**7)

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
        print pValue                                                        # DELETE
        pValue = pValue.to_integral_exact(rounding=ROUND_CEILING)           # WHAT'S UP WITH THIS FUNCTION?
        print pValue                                                        # DELETE
        pValue = int(pValue)

        return set[pValue - 1]

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

def trainSplitList(season, minTemps, meanTemps, maxTemps, weekdays, mortality):
    """

    :param season: int (0 is winter, 1 is summer)
    :param minTemps: 2D array containng minTemps, perhaps with lists of different lags
    :param meanTemps: 2D array containing meanTemps, " " " " " "
    :param maxTemps: 2D array containing maxTemps, " " " " " "
    :param mortality: list of mortality
    :param weekdays: list of weekdays 1-7
    :return:
    """

    #find number of temp lists
    numMinLists = len(minTemps)
    numMeanLists = len(meanTemps)
    numMaxLists = len(maxTemps)
    numMeasures = numMinLists + numMeanLists + numMaxLists + 1          # +2 accounts for weekdays and mortality

   #list len all checked by hand

    #find number of total days and 80% and 20% of days
                                                                                    # FIX: add something here to check that all the lists are the same len
    numDays = len(minTemps[0])                                                      # FIX: use above ^ instead of len(minTemps[0])

    numEightyPerc = int(math.ceil(0.8 * numDays))  # 940
    numTwentyPerc = int(0.2 * numDays)  # 234

    #list len all checked by hand

    #initialize
    pActualMort = []
    predictMort = []
    trainMeasures = []
    testMeasures = []
    coeffs = []
    intercepts = []
    error = []
    fStart = fEnd = listCount = pStart = pEnd = 0

    for i in range(numMeasures):
        trainMeasures.append([])
        testMeasures.append([])
        coeffs.append([])

    for i in range(numTwentyPerc + 1):
        regr = linear_model.LinearRegression()

        # fit
        fStart = i
        fEnd = numDays - 1 - numTwentyPerc + i

        #add temp and weekday data to trainMeasures
        for j in range(numMinLists):
            trainMeasures[j] = minTemps[j][fStart:fEnd + 1]
        listCount = numMinLists

        for j in range(listCount, listCount + numMeanLists):
            trainMeasures[j] = meanTemps[j-listCount][fStart:fEnd + 1]
        listCount = listCount + numMeanLists

        for j in range(listCount, listCount + numMaxLists):
            trainMeasures[j] = maxTemps[j-listCount][fStart:fEnd + 1]
        listCount = listCount + numMaxLists

        trainMeasures[listCount] = weekdays[fStart:fEnd + 1]
        listCount = 0

        #fit
        regr.fit((np.transpose(trainMeasures)).reshape(numEightyPerc, numMeasures), (np.transpose(mortality[fStart:fEnd + 1])).reshape(numEightyPerc,1))

        #gather regr coefficients and intercepts
        for j in range(numMeasures):
            coeffs[j].append(regr.coef_[0][j])
            intercepts.append(regr.intercept_[0])

        pStart = fEnd + 1
        pEnd = numDays - 1

        # add temp and weekday data to testMeasures
        for j in range(numMinLists):
            testMeasures[j] = minTemps[j][pStart:pEnd + 1]
        listCount = numMinLists

        for j in range(listCount, listCount + numMeanLists):
            testMeasures[j] = meanTemps[j - listCount][pStart:pEnd + 1]
        listCount = listCount + numMeanLists

        for j in range(listCount, listCount + numMaxLists):
            testMeasures[j] = maxTemps[j - listCount][pStart:pEnd + 1]
        listCount = listCount + numMaxLists

        testMeasures[listCount] = weekdays[pStart:pEnd + 1]

        #fill actual mortality to compare with predicted values
        pActualMort = mortality[pStart:pEnd + 1]

        #reset pStart and pEnd values
        pStart = 0
        pEnd = i - 1

        # add temp and weekday data to testMeasures
        for j in range(numMinLists):
            testMeasures[j] = list(testMeasures[j] + minTemps[j][pStart:pEnd + 1])
        listCount = numMinLists

        for j in range(listCount, listCount + numMeanLists):
            testMeasures[j] = list(testMeasures[j] + meanTemps[j - listCount][pStart:pEnd + 1])
        listCount = listCount + numMeanLists

        for j in range(listCount, listCount + numMaxLists):
            testMeasures[j] = list(testMeasures[j] + maxTemps[j - listCount][pStart:pEnd + 1])
        listCount = listCount + numMaxLists

        testMeasures[listCount] = list(testMeasures[listCount] + weekdays[pStart:pEnd + 1])

        # fill actual mortality to compare with predicted values
        pActualMort = list(pActualMort + mortality[pStart:pEnd + 1])

        # predict values
        predictMort = regr.predict((np.transpose(testMeasures)).reshape(numTwentyPerc, numMeasures))
        predictMort = (np.transpose(predictMort)[0].tolist())

        # calculate error
        for i in range(len(pActualMort)):
            error.append(np.mean((pActualMort[i] - predictMort[i])**2))

        plt.scatter(testMeasures[numMeasures-3], pActualMort)                                               # CHANGE
        plt.scatter(testMeasures[numMeasures-3], predictMort, color='red')                                  # CHANGE
        plt.show()

    #average out coeff, intercepts, error
    for i in range(numMeasures):
        coeffs[i] = np.float64(sum(coeffs[i]))/len(coeffs[i])
    intercepts = sum(intercepts)/len(intercepts)
    error = sum(error)/len(error)

    print error
    print coeffs
    print intercepts
    print

def makeDailySeasonalList(startIndex, endIndex, month, origList):
    newList = [[],[]]
    tempList = []
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
            newList[seasonIndex] = list(newList[seasonIndex] + tempList)

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

# initialize
smoothMort = []
subSmoothMort = []
smoothMeanTemp5 = []
smoothMeanTemp4 = []
smoothMeanTemp3 = []
smoothMaxTemp5 = []
smoothMinTemp5 = []

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

# calc lowerVal for 80% before casting to floats
lowerVal = calcPercentile(Decimal('0.8'), smoothMeanTemp5[4:])

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

# separate out summer and winter

#initialize
startIndex = 0
endIndex = 0

dailyMins5 = [[], []]                     # 0th index is winter
dailyMeans5 = [[],[]]
dailyMeans4 = [[],[]]
dailyMeans3 = [[],[]]
dailyMaxs5 = [[], []]

dailyUnsmthMins = [[],[]]
dailyUnsmthMeans = [[],[]]
dailyUnsmthMaxs = [[],[]]

dailySubSmoothMort = [[], []]
dailySubMeanMort = [[], []]             # note that the seasonal avg mort is subtracted,
                                        # NOT the avg mort over the ENTIRE time span
dailyUnsmthMort = [[], []]
dailyWeekday = [[],[]]

tempMin = []
tempUnsmthMin = []
tempMean = [[], [], []]               # 3 4 5
tempUnsmthMean = []
tempMax = []
tempUnsmthMax = []

tempSubSmoothMort = []
tempSubMeanMort = []
tempUnsmthMort = []

tempWeekday = []

# find first season of first year and last season of last year
for i in range(len(day)):
    if year[i] == 1987 and (month[i] < 6):
        startIndex = i + 1
    if year[i] == 2000 and month[i] == 12:
        endIndex = i - 1
        break
    # start and end indeces are correct (151 and 5082)

# loop through dates and mortality to separate into summer & winter
index = startIndex                                                  # CHANGE AS DESIRED
currentSeason = currentMonth = 0

while index <= endIndex:
    currentSeason = month[index]
    currentMonth = month[index]

    # iterate through a season
    while(sameSeason(currentSeason, currentMonth)) and index < len(year):
        currentMonth = month[index]

        #add to temp lists
        tempUnsmthMin.append(minTemps[index])
        tempUnsmthMean.append(meanTemps[index])
        tempUnsmthMax.append(maxTemps[index])
        tempMin.append(smoothMinTemp5[index])
        tempMean[0].append(smoothMeanTemp3[index])
        tempMean[1].append(smoothMeanTemp4[index])
        tempMean[2].append(smoothMeanTemp5[index])
        tempMax.append(smoothMaxTemp5[index])
        tempSubSmoothMort.append(subSmoothMort[index])
        tempSubMeanMort.append(mortality[index])
        tempWeekday.append(weekday[index])
        tempUnsmthMort.append(mortality[index])

        #update index and previousMonth
        index += 1
        if index <len(year):
            currentMonth = month[index]

    seasonIndex = calcSeasonModified(currentSeason)
    if seasonIndex < 3:
        dailyMins5[seasonIndex] = list(dailyMins5[seasonIndex] + tempMin)
        dailyMeans5[seasonIndex] = list(dailyMeans5[seasonIndex] + tempMean[2])
        dailyMeans4[seasonIndex] = list(dailyMeans4[seasonIndex] + tempMean[1])
        dailyMeans3[seasonIndex] = list(dailyMeans3[seasonIndex] + tempMean[0])
        dailyMaxs5[seasonIndex] = list(dailyMaxs5[seasonIndex] + tempMax)
        dailySubSmoothMort[seasonIndex] = list(dailySubSmoothMort[seasonIndex] + tempSubSmoothMort)
        dailySubMeanMort[seasonIndex] = list(dailySubMeanMort[seasonIndex] + tempSubMeanMort)
        dailyWeekday[seasonIndex] = list(dailyWeekday[seasonIndex] + tempWeekday)
        dailyUnsmthMort[seasonIndex] = list(dailyUnsmthMort[seasonIndex] + tempUnsmthMort)
        dailyUnsmthMins[seasonIndex] = list(dailyUnsmthMins[seasonIndex] + tempUnsmthMin)
        dailyUnsmthMeans[seasonIndex] = list(dailyUnsmthMeans[seasonIndex] + tempUnsmthMean)
        dailyUnsmthMaxs[seasonIndex] = list(dailyUnsmthMaxs[seasonIndex] + tempUnsmthMax)

    # clear temp lists
    tempMin = []
    tempMean = [[], [], []]
    tempMax = []
    tempSubSmoothMort = []
    tempSubMeanMort = []
    tempWeekday = []
    tempUnsmthMort = []
    tempUnsmthMin = []
    tempUnsmthMean = []
    tempUnsmthMax =[]

    #dailyUnsmthMort[0]/[1] and dailySubSmoothmort[0]/[1] are correct (excel sheet)

# subtract off values for tempSubMeanMort
summerMortAvg = np.float64(sum(dailySubMeanMort[1])) /len(dailySubMeanMort[1])
winterMortAvg = np.float64(sum(dailySubMeanMort[0]))/len(dailySubMeanMort[0])

for j in range(len(dailySubMeanMort[1])):
    dailySubMeanMort[1][j] = dailySubMeanMort[1][j] - summerMortAvg

for j in range(len(dailySubMeanMort[0])):
    dailySubMeanMort[0][j] = dailySubMeanMort[0][j] - winterMortAvg


"""
    Compare smoothed (lag of 5) and unsmoothed temperatures
    using subSmoothMortality for summer
"""
minParam = []
meanParam = []
maxParam = []

minParam.append(dailyUnsmthMins[1])
meanParam.append(dailyUnsmthMeans[1])
maxParam.append(dailyUnsmthMaxs[1])

#trainSplitList(0, minParam, meanParam, maxParam, dailyWeekday[1], dailySubSmoothMort[1])

minParam = []
meanParam = []
maxParam = []

minParam.append(dailyMins5[1])
meanParam.append(dailyMeans5[1])
maxParam.append(dailyMaxs5[1])

#trainSplitList(0, minParam, meanParam, maxParam, dailyWeekday[1], dailySubSmoothMort[1])

"""
    Use a shorter time span

"""
# find start and end indices
for i in range(len(day)):
    if year[i] == 1987 and (month[i] < 6):
        startIndex = i + 1
    if year[i] == 1993 and month[i] == 12:
        endIndex = i - 1
        break
    # indices are correct (151 and 2525)

#initialize
dailyUnsmthMins = [[],[]]
dailyUnsmthMeans = [[],[]]
dailyUnsmthMaxs = [[],[]]

dailyMins5 = [[],[]]
dailyMeans5 = [[],[]]
dailyMaxs5 =[[],[]]

dailySubMeanMort = [[], []]             # note that the seasonal avg mort is subtracted,
                                        # NOT the avg mort over the ENTIRE time span
dailyWeekday = [[],[]]

# get seasonal data
dailyUnsmthMins = makeDailySeasonalList(startIndex, endIndex, month, minTemps)
dailyUnsmthMeans = makeDailySeasonalList(startIndex, endIndex, month, meanTemps)
dailyUnsmthMaxs = makeDailySeasonalList(startIndex, endIndex, month, maxTemps)
dailyMins5 = makeDailySeasonalList(startIndex, endIndex, month, smoothMinTemp5)
dailyMeans5 = makeDailySeasonalList(startIndex, endIndex, month, smoothMeanTemp5)
dailyMaxs5 = makeDailySeasonalList(startIndex, endIndex, month, smoothMaxTemp5)
dailyWeekday = makeDailySeasonalList(startIndex, endIndex, month, weekday)
dailySubMeanMort = makeDailySeasonalList(startIndex, endIndex, month, mortality)

summerMortAvg = np.float64(sum(dailySubMeanMort[1])) /len(dailySubMeanMort[1])
winterMortAvg = np.float64(sum(dailySubMeanMort[0]))/len(dailySubMeanMort[0])

for j in range(len(dailySubMeanMort[1])):
    dailySubMeanMort[1][j] = dailySubMeanMort[1][j] - summerMortAvg

for j in range(len(dailySubMeanMort[0])):
    dailySubMeanMort[0][j] = dailySubMeanMort[0][j] - winterMortAvg


# run split lists
minParam = []
meanParam = []
maxParam = []

minParam.append(dailyUnsmthMins[1])
meanParam.append(dailyUnsmthMeans[1])
maxParam.append(dailyUnsmthMaxs[1])

#trainSplitList(0, minParam, meanParam, maxParam, dailyWeekday[1], dailySubMeanMort[1])

minParam = []
meanParam = []
maxParam = []

minParam.append(dailyMins5[1])
meanParam.append(dailyMeans5[1])
maxParam.append(dailyMaxs5[1])

trainSplitList(0, minParam, meanParam, maxParam, dailyWeekday[1], dailyUnsmthMort[1])