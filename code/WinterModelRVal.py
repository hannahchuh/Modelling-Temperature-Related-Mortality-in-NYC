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

# read in future data (mislabeled "hist")
modelName = "gfdl"

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

fileName = modelName + "HistCompiled.csv"
backDict = pickle.load(open(fileName, 'rb'))
backTemp = backDict['meanTemps']
backDewPt = backDict['dewPts']
backMonth = backDict['month']
backYear = backDict['year']
backTep = celsiusToFahrenheit(backTemp)
backDewPt =[i+np.float64(273.15) for i in backDewPt]
backDewPt = celsiusToFahrenheit(backDewPt)
bSmoothMeanTemp5 = rollingAvg(5, [Decimal(i) for i in backTemp])
bSmoothMeanTemp5 = [np.float64(i) for i in bSmoothMeanTemp5]

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

dailyBackMeans5 = makeYearlySeasonalList(startIndex, endIndex-4, numWinterYears, backMonth, bSmoothMeanTemp5)       # -4 to account for missing leap years
dailyBackDewPts = makeYearlySeasonalList(startIndex, endIndex-4, numWinterYears, backMonth, backDewPt)

# start and end indices for 2020-2080
for i in range(len(histMonth)):
    if histYear[i] == 2020 and (histMonth[i] < 6):
        startIndex = i + 1
    if histYear[i] == 2080 and histMonth[i] == 12:
        endIndex = i - 1
        break
numWinterYears = 2080 - 2020

dailyHistMeans5 = makeYearlySeasonalList(startIndex, endIndex, numWinterYears, histMonth, hSmoothMeanTemp5)
dailyHistDewPts = makeYearlySeasonalList(startIndex, endIndex, numWinterYears, histMonth, histDewPt)

#make year perc list
dailyYear = [[], []]
dailyHistYear = [[], []]
dailyBackYear = [[],[]]

startingYear = 1988             # starting off at winter
for i in range(len(dailyMeans5)):
    loopingYear = startingYear
    for j in range(len(dailyMeans5[i])):
        dailyYear[i].append([])
        for k in range(len(dailyMeans5[i][j])):
            dailyYear[i][j].append(loopingYear)
        loopingYear += 1
    startingYear -= 1

startingYear = 1988
for i in range(len(dailyBackMeans5)):
    loopingYear = startingYear
    for j in range(len(dailyBackMeans5[i])):
        dailyBackYear[i].append([])
        for k in range(len(dailyBackMeans5[i][j])):
            dailyBackYear[i][j].append(loopingYear)
        loopingYear += 1
    startingYear -= 1

startingYear = 2021             # starting off at winter

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
bYearlyPercentiles = [[],[]]

for i in range(len(dailyMeans5)):
    for j in range(len(dailyMeans5[i])):
        yearlyPercentiles[i].append(calcPercentile(percent, dailyMeans5[i][j]))

for i in range(len(dailyBackMeans5)):
    for j in range(len(dailyBackMeans5[i])):
        bYearlyPercentiles[i].append(calcPercentile(percent, dailyBackMeans5[i][j]))

for i in range(len(dailyHistMeans5)):
    for j in range(len(dailyHistMeans5[i])):
        hYearlyPercentiles[i].append(calcPercentile(percent, dailyHistMeans5[i][j]))

# specify winter or summer
season = 0

# initialize
tempPercList = []
mortPercList = []
dewPercList = []
yearPercList = []

histPredictMort = []
histTempPercList = []
histDewPercList = []
histYearPercList = []

backPredictMort = []
backTempPercList = []
backDewPercList = []
backYearPercList = []

tempCopy = []
dewCopy = []
yearCopy = []
mortCopy = []

#append all seasonal lists into the perc list (either seasonal threshold or no threshold)
for i in range(len(dailyMeans5[season])):
    tempPercList = list(tempPercList + dailyMeans5[season][i])
    dewPercList= list(dewPercList + dailyDewPts[season][i])
    mortPercList = list(mortPercList + dailySubSmoothMort[season][i])
    yearPercList = list(yearPercList + dailyYear[season][i])

for i in range(len(dailyBackMeans5[season])):
    backTempPercList = list(backTempPercList + dailyBackMeans5[season][i])
    backDewPercList = list(backDewPercList + dailyBackDewPts[season][i])
    backYearPercList = list(backYearPercList + dailyBackYear[season][i])

for i in range(len(dailyHistMeans5[season])):
    histTempPercList= list(histTempPercList+ dailyHistMeans5[season][i])
    histDewPercList= list(histDewPercList+ dailyHistDewPts[season][i])
    histYearPercList = list(histYearPercList + dailyHistYear[season][i])


# initialize for model
actualMort = []
predictMort = []
mortTrain = []
trainMeasures = []
testMeasures = []
backMeasures = []
histMeasures = []
coeffs = []
intercepts = []
fStart = fEnd = listCount = pStart = pEnd = 0

numMeasures = 2
numDays = len(tempPercList)

for i in range(numMeasures):
    trainMeasures.append([])
    histMeasures.append([])
    backMeasures.append([])
    testMeasures.append([])
    coeffs.append([])

histMeasures[0] = histTempPercList
histMeasures[1] = histDewPercList

backMeasures[0] = backTempPercList
backMeasures[1] = backDewPercList

regr = linear_model.LinearRegression()

trainMeasures[0] = tempPercList
trainMeasures[1] = dewPercList
mortTrain = mortPercList

# fit
#regr.fit((np.transpose(trainMeasures)).reshape(numDays, numMeasures), (np.transpose(mortTrain)).reshape(numDays, 1))



#starting r value calculation
numDays = len(tempPercList) # 1174
numEightyPerc = int(math.ceil(0.8 * numDays))  #940
numTwentyPerc = int(0.2 * numDays)  # 234
numMeasures = 2

# list len all checked by hand

# initialize

fStart = fEnd = listCount = pStart = pEnd = 0

rTotal = 0

for i in range(numTwentyPerc + 1):
    regr = linear_model.LinearRegression()

    # fit
    fStart = i
    fEnd = numDays - 1 - numTwentyPerc + i

    #print fStart, fEnd+1, len(trainMeasures[0])
    trainMeasures[0] = tempPercList[fStart:fEnd+1]
    trainMeasures[1] = dewPercList[fStart:fEnd+1]

    # fit
    regr.fit((np.transpose(trainMeasures)).reshape(numEightyPerc, numMeasures), (np.transpose(mortPercList[fStart:fEnd + 1])).reshape(numEightyPerc, 1))
    plt.scatter(trainMeasures[0], mortPercList[fStart:fEnd+1])
    #plt.show()

    # gather regr coefficients and intercepts
    for j in range(numMeasures):
        coeffs[j].append(regr.coef_[0][j])
    #print regr.intercept_
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
    pActualMort = mortPercList[pStart:pEnd + 1]

    # reset pStart and pEnd values
    pStart = 0
    pEnd = i - 1


    #print pStart, pEnd,

    # add temp and weekday data to testMeasures
    testMeasures[0] = list(testMeasures[0] + tempPercList[pStart:pEnd+1])
    testMeasures[1] = list(testMeasures[1] + dewPercList[pStart:pEnd+1])

    # fill actual mortality to compare with predicted values
    pActualMort = list(pActualMort + mortPercList[pStart:pEnd + 1])

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
print "r avg is: " + str(rTotal/(numTwentyPerc))
print "total data points are: " + str(numEightyPerc)


#end r vlaue calucation






# gather regr coefficients and intercepts
for j in range(numMeasures):
    coeffs[j].append(regr.coef_[0][j])
intercepts = regr.intercept_[0]


# add temp and weekday data to testMeasures
testMeasures[0] = tempPercList
testMeasures[1] = dewPercList

# fill actual mortality to compare with predicted values
actualMort = mortPercList

# predict values
predictMort = regr.predict((np.transpose(trainMeasures)).reshape(numDays, numMeasures))
predictMort = (np.transpose(predictMort)[0].tolist())

#calculating score
print len(trainMeasures[0])
print len(trainMeasures[1])
print len(actualMort)
print len(predictMort)
print regr.score((np.transpose(trainMeasures)).reshape(numDays, numMeasures), actualMort)
rms = 0;
for i in range(len(predictMort)):
    rms += ((predictMort[i]-actualMort[i])**2)
print "RMS IS: " + str(np.sqrt(rms/len(predictMort)))

# predict values historical data
histPredictMort = regr.predict((np.transpose(histMeasures)).reshape(len(histMeasures[0]), numMeasures))
histPredictMort = (np.transpose(histPredictMort)[0].tolist())

# predict values - back data
backPredictMort = regr.predict((np.transpose(backMeasures)).reshape(len(backMeasures[0]), numMeasures))
backPredictMort = (np.transpose(backPredictMort)[0].tolist())

plt.scatter(testMeasures[0], actualMort)
plt.scatter(testMeasures[0], predictMort, color='red')
plt.scatter(backMeasures[0], backPredictMort, color='black')
plt.scatter(histMeasures[0], histPredictMort, color='green')
#plt.show()

print coeffs
print intercepts

#looking at mortality per year
hYearlyMort = splitMortYearlyNan(histYearPercList, histPredictMort, season)
pYearlyMort = splitMortYearlyNan(yearPercList, predictMort, season)
bYearlyMort = splitMortYearlyNan(backYearPercList, backPredictMort, season)

for i in range(len(bYearlyMort)):
    if bYearlyMort[i] == ['nan']:
        bYearlyMort[i] = 'nan'
    else:
        bYearlyMort[i] = sum(bYearlyMort[i])

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

# plot mortality sum per year
plt.hold(True)
if season == 0:
    startingYear = 1988
else:
    startingYear = 1987
xAxis = np.arange(startingYear,2001)
plt.scatter(xAxis, bYearlyMort, color = 'blue')
if len(hYearlyMort) > 14:
    if season == 0:
        xAxis = np.arange(2021, 2081)
    else:
        xAxis = np.arange(2020,2081)
plt.scatter(xAxis, hYearlyMort, color = 'blue')
plt.title("Yearly sum of winter mortality anomalies (GFDL-CM3G)", fontsize =15)
plt.xlabel("Year", fontsize=15)
plt.ylabel("Total mortality anomaly", fontsize=15)
#plt.show()


"""
# pickle future mortality proj
exportDict = {}
fileName = modelName + "ModelMort"
if season == 0:
    fileName += "Winter.csv"
else:
    fileName += "Summer.csv"
exportDict.update({'FutureMortality':hYearlyMort})
exportDict.update({'HistMortality':bYearlyMort})
with open(fileName, 'wb') as handle:
    pickle.dump(exportDict, handle)

# write future mortality proj to readable csv file
sampleDict = exportDict
dictLength = len(sampleDict)
tempList = []
fileName = modelName + "ModelMortReadable"
if season == 0:
    fileName += "Winter.csv"
else:
    fileName += "Summer.csv"
with open(fileName, "wb") as fileObj:
    fileWriter = csv.writer(fileObj)
    listLength = len(sampleDict.itervalues().next())
    for index in range(listLength):
        for key in sampleDict:
            if index < len(sampleDict[key]):
                tempList.append(sampleDict[key][index])
        fileWriter.writerow(tempList)
        tempList = []
"""