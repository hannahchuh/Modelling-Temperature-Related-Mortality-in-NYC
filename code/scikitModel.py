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
        intercepts = intercepts + regr.intercept_

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

        #fill actual mortality to compare with predicted values
        pActualMort = list(pActualMort + mortality[pStart:pEnd + 1])

        #predict values
        predictMort = regr.predict((np.transpose(testMeasures)).reshape(numTwentyPerc, numMeasures))
        predictMort = (np.transpose(predictMort)[0].tolist())

        plt.scatter(testMeasures[numMeasures-3], pActualMort)
        plt.scatter(testMeasures[numMeasures-3], predictMort, color='red')
        #plt.show()

    for i in range(numMeasures):
        coeffs[i] = np.float64(sum(coeffs[i]))/len(coeffs[i])

    print coeffs
    print intercepts

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
smoothedMortality = []
subSmoothMortality = []
smoothMeanTemp5 = []
smoothMeanTemp4 = []
smoothMeanTemp3 = []
smoothMaxTemp5 = []
smoothMinTemp5 = []

# smooth temperature set
smoothedMortality = rollingAvg(30, mortality)
smoothMeanTemp5 = rollingAvg(5, meanTemps)                               # change this as desired
smoothMeanTemp4 = rollingAvg(4, meanTemps)
smoothMeanTemp3 = rollingAvg(3, meanTemps)
smoothMinTemp5 = rollingAvg(5, minTemps)
smoothMaxTemp5 = rollingAvg(5, maxTemps)

# create subSmoothMort list
for i in range(len(smoothedMortality)):
    if smoothedMortality[i] == Decimal('nan'):
        subSmoothMortality.append(Decimal('nan'))
    else:
        subSmoothMortality.append(Decimal(mortality[i] - smoothedMortality[i]))

#print subSmoothMort                                            # DELETE
#print smoothMeanTemp5                                                  # DELETE
#print smoothMort                                             # DELETE

# calc lowerVal for 80% before casting to floats
lowerVal = calcPercentile(Decimal('0.9'), smoothMeanTemp5[4:])

# cast as floats
for i in range(len(smoothedMortality)):
    smoothMeanTemp5[i] = np.float64(smoothMeanTemp5[i])
    smoothMeanTemp4[i] = np.float64(smoothMeanTemp4[i])
    smoothMeanTemp3[i] = np.float64(smoothMeanTemp3[i])
    smoothMinTemp5[i] = np.float64(smoothMinTemp5[i])
    smoothMaxTemp5[i] = np.float64(smoothMaxTemp5[i])
    subSmoothMortality[i] = np.float64(subSmoothMortality[i])
    meanTemps[i] = np.float64(meanTemps[i])
    minTemps[i] = np.float64(minTemps[i])
    maxTemps[i] = np.float64(maxTemps[i])
    smoothedMortality[i] = np.float64(smoothedMortality[i])          # DELETE

#print smoothMeanTemp5                                               # DELETE
#print smoothMort                                             # DELETE

#VARIOUS ORDER FIT
'''
# fit line
popt, pcov = opt.curve_fit(sixthOrder, smoothMeanTemp5[29:], subSmoothMort[29:])
print popt
line = []
count = 0
for i in range(29, len(smoothMeanTemp5)):
    tempVal = smoothMeanTemp5[i]
    #yVal = popt[0] + popt[1]*tempVal + popt[2]*tempVal*tempVal + popt[3]*tempVal*tempVal*tempVal + popt[4]*tempVal*tempVal*tempVal*tempVal
    #if
    yVal = sixthOrder(smoothMeanTemp5[i], popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])
    line.append(yVal)
    count += 1

print "count is: " + str(count)
plt.plot(smoothMeanTemp5[29:], line, 'ro')
plt.scatter(smoothMeanTemp5[29:], subSmoothMort[29:])
plt.show()
'''

'''
#LINEAR FIT WITH 80% AND UP
# x and y axis
xLine = []
yLine = []

# plot line and scatter plot
print lowerVal
for i in range(29, len(subSmoothMort)):
    if smoothMeanTemp5[i] > lowerVal:
        xLine.append(smoothMeanTemp5[i])
        yLine.append(subSmoothMort[i])
print xLine
print yLine

#plt.scatter(smoothMeanTemp5, subSmoothMort)
line = []
slope, intercept, r_value, p_value, std_err = stats.linregress(xLine, yLine)
print slope
for i in range(len(xLine)):
    line.append((np.float64(slope*xLine[i]))+intercept)
plt.plot(xLine, line)
plt.scatter(xLine, yLine)
plt.show()
'''

'''
# PRINT LINEAR FIT FOR smoothMeanTemp5 AND subSmoothMort WITH LINREGRESS

slope, intercept, r_value, p_value, std_err = stats.linregress(smoothMeanTemp5, subSmoothMort)
print "SLOPE IS: " + str(slope)
line = []
for i in range(29, len(smoothMeanTemp5)):
    line.append(np.float64(slope*smoothMeanTemp5[i])+intercept)
plt.plot(smoothMeanTemp5, line)
'''

'''
# SCIKIT LINEAR REGRESSION - ALL SEASONS
regr = linear_model.LinearRegression()
regr.fit((np.transpose(smoothMeanTemp5[29:])).reshape(5085,1), (np.transpose(subSmoothMort[29:])).reshape(5085,1))
print regr.coef_
print regr.intercept_

line = []
for i in range(29, len(smoothMeanTemp5)):
    line.append(np.float64(regr.coef_*smoothMeanTemp5[i])+regr.intercept_)
plt.plot(smoothMeanTemp5[29:], line)
plt.scatter(smoothMeanTemp5[29:], subSmoothMort[29:])
plt.show()
'''

# separate out summer and winter

#initialize
startIndex = 0
endIndex = 0
dailyMins = [[],[]]                     # 0th index is winter
dailyMeans5 = [[],[]]
dailyMeans4 = [[],[]]
dailyMeans3 = [[],[]]
dailyMaxs = [[],[]]
dailyUnsmthMins = [[],[]]
dailyUnsmthMeans = [[],[]]
dailyUnsmthMaxs = [[],[]]
dailyMort = [[], []]
dailyMortMean = [[],[]]
dailyUnsmoothedMort = [[],[]]
dailyWeekday = [[],[]]
tempMinList = []
tempMinUnsmth = []
tempMeanList = [[],[],[]]               # 3 4 5
tempMeanUnsmth = []
tempMaxList = []
tempMaxUnsmth = []
tempMortList = []
tempMortMeanList = []
tempUnsmthMort = []
tempDayList = []
winterSlopes = [[],[],[],[],[],[]]                   # 0th index is slope, 1st index is intercept
winterIntcp = 0
summerCoeff = [[],[]]
fitStart = 0
fitEnd = 0
predictStart = 0
predictEnd = 0
predictTemp = []
predictWeekday= []
pActualMortality = []
predictMortality = []
weekDayMortality = [[[],[],[],[],[],[],[]], [[],[],[],[],[],[],[]]]

#find 1st season of 1st year and last season of last year
for i in range(len(day)):
    if year[i] == 1987 and (month[i] < 6):
        startIndex = i + 1
    if year[i] == 2000 and month[i] == 12:
        endIndex = i - 1
        break
#start index and end index correct (151 and 5082)

#loop through dates and mortality to separate into summer & winter
index = startIndex                                                  # CHANGE AS DESIRED
currentSeason = currentMonth = 0

while index <= endIndex:
    currentSeason = month[index]
    currentMonth = month[index]

    #diterate through a season
    while(sameSeason(currentSeason, currentMonth)) and index < len(year):
        currentMonth = month[index]

        #add to temp lists
        tempMinUnsmth.append(minTemps[index])
        tempMeanUnsmth.append(meanTemps[index])
        tempMaxUnsmth.append(meanTemps[index])
        tempMinList.append(smoothMinTemp5[index])
        tempMeanList[0].append(smoothMeanTemp3[index])
        tempMeanList[1].append(smoothMeanTemp4[index])
        tempMeanList[2].append(smoothMeanTemp5[index])
        tempMaxList.append(smoothMaxTemp5[index])
        tempMortList.append(subSmoothMortality[index])
        tempDayList.append(weekday[index])
        tempUnsmthMort.append(mortality[index])

        #update index and previousMonth
        index += 1
        if index <len(year):
            currentMonth = month[index]

    seasonIndex = calcSeasonModified(currentSeason)
    if seasonIndex < 3:
        dailyMins[seasonIndex] = list( dailyMins[seasonIndex] + tempMinList )
        dailyMeans5[seasonIndex] = list(dailyMeans5[seasonIndex] + tempMeanList[2])
        dailyMeans4[seasonIndex] = list(dailyMeans4[seasonIndex] + tempMeanList[1])
        dailyMeans3[seasonIndex] = list(dailyMeans3[seasonIndex] + tempMeanList[0])
        dailyMaxs[seasonIndex] = list( dailyMaxs[seasonIndex] + tempMaxList )
        dailyMort[seasonIndex] = list(dailyMort[seasonIndex] + tempMortList)
        dailyWeekday[seasonIndex] = list( dailyWeekday[seasonIndex] + tempDayList)
        dailyUnsmoothedMort[seasonIndex] = list(dailyUnsmoothedMort[seasonIndex] + tempUnsmthMort)
        dailyUnsmthMins[seasonIndex] = list(dailyUnsmthMins[seasonIndex] + tempMinUnsmth )
        dailyUnsmthMeans[seasonIndex] = list(dailyUnsmthMeans[seasonIndex] + tempMeanUnsmth )
        dailyUnsmthMaxs[seasonIndex] = list(dailyUnsmthMaxs[seasonIndex] + tempMaxUnsmth)

    #clear temp lists
    tempMinList = []
    tempMeanList = [[],[],[]]
    tempMaxList = []
    tempMortList = []
    tempDayList = []
    tempUnsmthMort = []
    tempMinUnsmth = []
    tempMeanUnsmth = []
    tempMaxUnsmth =[]


"""
#TESTING TESTING TESTING TESTING
#find new start and end values for 1987-1993
for i in range(len(day)):
    if year[i] == 1987 and (month[i] < 6):
        startIndex = i + 1
    if year[i] == 1993 and month[i] == 12:
        endIndex = i - 1
        break
#start index and end index correct (151 and 5082)

dailyMins5 = [[],[]]
dailyMeans3= [[],[]]
dailyMeans4= [[],[]]
dailyMeans5 = [[],[]]
dailyMaxs5 = [[],[]]

#loop through dates and mortality to separate into summer & winter
index = startIndex                                                  # CHANGE AS DESIRED
currentSeason = currentMonth = 0

while index <= endIndex:
    currentSeason = month[index]
    currentMonth = month[index]

    #diterate through a season
    while(sameSeason(currentSeason, currentMonth)) and index < len(year):
        currentMonth = month[index]

        #add to temp lists
        tempMin.append(smoothMinTemp5[index])
        tempMean[0].append(smoothMeanTemp3[index])
        tempMean[1].append(smoothMeanTemp4[index])
        tempMean[2].append(smoothMeanTemp5[index])
        tempMax.append(smoothMaxTemp5[index])
        tempSubMeanMort.append(mortality[index])
        tempWeekday.append(weekday[index])

        #update index and previousMonth
        index += 1
        if index <len(year):
            currentMonth = month[index]

    seasonIndex = calcSeasonModified(currentSeason)
    if seasonIndex < 3:
        dailyMins5[seasonIndex] = list( dailyMins5[seasonIndex] + tempMin )
        dailyMeans5[seasonIndex] = list(dailyMeans5[seasonIndex] + tempMean[2])
        dailyMeans4[seasonIndex] = list(dailyMeans4[seasonIndex] + tempMean[1])
        dailyMeans3[seasonIndex] = list(dailyMeans3[seasonIndex] + tempMean[0])
        dailyMaxs5[seasonIndex] = list( dailyMaxs5[seasonIndex] + tempMax )
        dailySubMeanMort[seasonIndex] = list(dailySubMeanMort[seasonIndex] + tempSubMeanMort)
        dailyWeekday[seasonIndex] = list( dailyWeekday[seasonIndex] + tempWeekday)

    #clear temp lists
    tempMin = []
    tempMean = [[],[],[]]
    tempMax = []
    tempSubMeanMort = []
    tempWeekday = []

#TESTING TESTING TESTING TESTING
"""


"""
The following uncommented code will create an error, as it was being used to test
dailySubMeanMort for a shorter time frame (1987-1993). The above commented code "TESTING TESTING.."
should be uncommented for the below code to work.
"""

#subtract off values for tempSubMeanMort
summerMortAvg = np.float64(sum(dailyMortMean[1]))/len(dailyMortMean[1])
for j in range(len(dailyMortMean[1])):
    dailyMortMean[1][j] = dailyMortMean[1][j] - summerMortAvg

xAxis = np.arange(len(dailyMortMean[1]))
plt.plot(xAxis, dailyMortMean[1])
#plt.show()

#ST
season = 1
minParam = []
minParam.append(dailyMins[season])

meanParam = []
meanParam.append(dailyMeans3[season])
meanParam.append(dailyMeans4[season])
meanParam.append(dailyMeans5[season])

maxParam = []
maxParam.append(dailyMaxs[season])

#trainSplitList(season, minParam, meanParam, maxParam, dailyWeekday[season], dailyMortMean[season])
print "after...."
#ET


#ST
season = 1

minParam = []
minParam.append(dailyUnsmthMins[season])

meanParam = []
meanParam.append(dailyUnsmthMeans[season])

maxParam = []
maxParam.append(dailyUnsmthMaxs[season])

trainSplitList(season, minParam, meanParam, maxParam, dailyWeekday[season], dailySubSmoothMort[season])
#ET


"""


#len of dailyMins5/Means/Maxs' list is correct (JJA - 14, DJF - 13)          # RECHECK THIS

# SCIKIT LINREGRESS CYCLE THROUGH 80/20 FOR SUMMER W/ UNSMOOTHED TEMPS
# scikit linear regression - cycle through with 80/20 for winter
# CHANGE THIS SO INDECES ARE NOT HARD CODED
season = 1
totalLen = len(dailyMins5[season])
numEightyPer = int(math.ceil(0.8 * totalLen))  # 940
numTwentyPer = int(0.2 * totalLen)  # 234

xMatrix = [[], [], [], []]
xPredictMatrix = [[], [], [], []]
error = []
for i in range(numTwentyPer + 1):  # 80% can be shifted up 234 times
    regr = linear_model.LinearRegression()

    # fit
    fitStart = i
    fitEnd = totalLen - 1 - numTwentyPer + i

    xMatrix[0] = dailyUnsmthMins[season][fitStart:fitEnd + 1]
    xMatrix[1] = dailyUnsmthMeans[season][fitStart:fitEnd + 1]
    xMatrix[2] = dailyUnsmthMaxs[season][fitStart:fitEnd + 1]
    xMatrix[3] = dailyWeekday[season][fitStart:fitEnd + 1]
    regr.fit((np.transpose(xMatrix)).reshape(numEightyPer, 4),
             (np.transpose(dailySubSmoothMort[season][fitStart:fitEnd + 1])).reshape(numEightyPer, 1))

    for j in range(4):
        winterSlopes[j].append(regr.coef_[0][j])  # since regr.coef_ is in a nested array
    winterIntcp = winterIntcp + (regr.intercept_)

    # predict
    predictStart = fitEnd + 1
    predictEnd = totalLen - 1

    xPredictMatrix[0] = dailyUnsmthMins[season][predictStart:predictEnd + 1]
    xPredictMatrix[1] = dailyUnsmthMeans[season][predictStart:predictEnd + 1]
    xPredictMatrix[2] = dailyUnsmthMaxs[season][predictStart:predictEnd + 1]
    xPredictMatrix[3] = dailyWeekday[season][predictStart:predictEnd + 1]
    pActualMortality = dailySubSmoothMort[season][predictStart:predictEnd + 1]
    predictStart = 0
    predictEnd = i - 1

    xPredictMatrix[0] = list(xPredictMatrix[0] + dailyUnsmthMins[season][predictStart:predictEnd + 1])
    xPredictMatrix[1] = list(xPredictMatrix[1] + dailyUnsmthMeans[season][predictStart:predictEnd + 1])
    xPredictMatrix[2] = list(xPredictMatrix[2] + dailyUnsmthMaxs[season][predictStart:predictEnd + 1])
    xPredictMatrix[3] = list(xPredictMatrix[3] + dailyWeekday[season][predictStart:predictEnd + 1])
    pActualMortality = list(pActualMortality + dailySubSmoothMort[season][predictStart:predictEnd + 1])

    predictMortality = regr.predict((np.transpose(xPredictMatrix)).reshape(numTwentyPer, 4))[0]
    # predictMortality = np.transpose(predictMortality)

    # print regr.score(predictMortality, np.transpose(pActualMortality).reshape(numTwentyPer,1))
    plt.plot(xPredictMatrix[1], regr.predict((np.transpose(xPredictMatrix)).reshape(numTwentyPer, 4)), 'ro')
    plt.scatter(xPredictMatrix[1], pActualMortality)
    #plt.show()

for i in range(4):
    winterSlopes[i] = np.float64(sum(winterSlopes[i])) / len(winterSlopes[i])
winterIntcp = np.float64(sum(winterIntcp)) / len(winterIntcp)

print winterSlopes
print winterIntcp

print error
print np.float64(sum(error)) / len(error)


"""

"""
# scikit lin regrss w/ multiple x values

# scikit linear regression - cycle through with 80/20 for winter
                                                                            # CHANGE THIS SO INDECES ARE NOT HARD CODED
season = 0
totalLen = len(dailyMins5[season])
numEightyPer = int(math.ceil(0.8*totalLen))    #940
numTwentyPer = int(0.2*totalLen)    #234

xMatrix = [[],[],[],[],[],[]]
xPredictMatrix = [[],[],[],[],[],[]]
error = []
for i in range(numTwentyPer + 1):                                #80% can be shifted up 234 times
    regr = linear_model.LinearRegression()

    #fit
    fitStart = i
    fitEnd = totalLen - 1 - numTwentyPer + i

    xMatrix[0] = dailyMins5[season][fitStart:fitEnd+1]
    xMatrix[1] = dailyMeans3[season][fitStart:fitEnd+1]
    xMatrix[2] = dailyMeans4[season][fitStart:fitEnd+1]
    xMatrix[3] = dailyMeans5[season][fitStart:fitEnd+1]
    xMatrix[4] = dailyMaxs5[season][fitStart:fitEnd+1]
    xMatrix[5] = dailyWeekday[season][fitStart:fitEnd+1]
    regr.fit((np.transpose(xMatrix)).reshape(numEightyPer,6), (np.transpose(dailySubSmoothMort[season][fitStart:fitEnd+1])).reshape(numEightyPer,1))

    for j in range(6):
        winterSlopes[j].append(regr.coef_[0][j])            #since regr.coef_ is in a nested array
    winterIntcp = winterIntcp + (regr.intercept_)

    #predict
    predictStart = fitEnd + 1
    predictEnd = totalLen - 1

    xPredictMatrix[0] = dailyMins5[season][predictStart:predictEnd+1]
    xPredictMatrix[1] = dailyMeans3[season][predictStart:predictEnd+1]
    xPredictMatrix[2] = dailyMeans4[season][predictStart:predictEnd+1]
    xPredictMatrix[3] = dailyMeans5[season][predictStart:predictEnd+1]
    xPredictMatrix[4] = dailyMaxs5[season][predictStart:predictEnd+1]
    xPredictMatrix[5] = dailyWeekday[season][predictStart:predictEnd+1]
    pActualMortality = dailySubSmoothMort[season][predictStart:predictEnd + 1]
    predictStart = 0
    predictEnd = i - 1

    xPredictMatrix[0] = list(xPredictMatrix[0] + dailyMins5[season][predictStart:predictEnd+1])
    xPredictMatrix[1] = list(xPredictMatrix[1] + dailyMeans3[season][predictStart:predictEnd+1])
    xPredictMatrix[2] = list(xPredictMatrix[2] + dailyMeans4[season][predictStart:predictEnd+1])
    xPredictMatrix[3] = list(xPredictMatrix[3] + dailyMeans5[season][predictStart:predictEnd+1])
    xPredictMatrix[4] = list(xPredictMatrix[4] + dailyMaxs5[season][predictStart:predictEnd+1])
    xPredictMatrix[5] = list(xPredictMatrix[5] + dailyWeekday[season][predictStart:predictEnd+1])
    pActualMortality = list(pActualMortality + dailySubSmoothMort[season][predictStart:predictEnd + 1])


    predictMortality = regr.predict((np.transpose(xPredictMatrix)).reshape(numTwentyPer,6))[0]
    #predictMortality = np.transpose(predictMortality)

    #print regr.score(predictMortality, np.transpose(pActualMortality).reshape(numTwentyPer,1))
    plt.plot(xPredictMatrix[3], regr.predict((np.transpose(xPredictMatrix)).reshape(numTwentyPer,6)), 'ro')
    plt.scatter(xPredictMatrix[3], pActualMortality)
    plt.show()


for i in range(6):
    winterSlopes[i] = np.float64(sum(winterSlopes[i]))/len(winterSlopes[i])
winterIntcp = np.float64(sum(winterIntcp))/len(winterIntcp)

print winterSlopes
print winterIntcp

print error
print np.float64(sum(error))/len(error)
"""

"""
xMatrix = [[],[]]
for i in range(258):                                #80% can be shifted up 234 times
    regr = linear_model.LinearRegression()

    #fit
    fitStart = i
    fitEnd = 1029 - 257 + i
    xMatrix[0] = tempList[1][fitStart:fitEnd + 1]
    xMatrix[1] = dailyWeekday[1][fitStart:fitEnd + 1]
    regr.fit((np.transpose(xMatrix)).reshape(773,2), (np.transpose(dailyMortality[1][fitStart:fitEnd+1])).reshape(773,1))

    summerCoeff[0].append(regr.coef_)
    summerCoeff[1].append(regr.intercept_)

    #comment below out
    #predict
    predictStart = fitEnd + 1
    predictEnd = 1029

    predictTemp = tempList[0][predictStart:predictEnd+1]
    predictWeekday = dailyWeek
    predictStart = 0
    predictEnd = i - 1

    predictTemp = list(predictTemp + tempList[1][predictStart:predictEnd+1])
    plt.plot(predictTemp, regr.predict((np.transpose(predictTemp)).reshape(257,1)), 'ro')
    plt.scatter(tempList[1], dailyMortality[1])
    #plt.show()
    # comment above out
"""


"""
# SCIKIT LINREGRESS W/ 8 YRS 2YRS SUMMER
yearlyMins = []
yearlyMeans = []
yearlyMaxs = []

index = startIndex
currentYear = year[startIndex]
currentSeason = month[index]
indexYear = year[startIndex]

while index <= endIndex:

    currentYear = year[index]
    currentSeason = month[index]
    indexYear = year[index]

    while(currentYear == indexYear and index <= endIndex):
        indexYear = year[index]
        tempMin.append(smoothMinTemp5[index])
        tempMean[0].append(smoothMeanTemp3[index])
        tempMean[1].append(smoothMeanTemp4[index])
        tempMean[2].append(smoothMinTemp5[index])
        tempMax.append(smoothMaxTemp5[index])
        tempMortalityList.append(subSmoothMort[index])
        tempWeekday.append(weekday[index])

        index += 1
        if index <= endIndex:
            indexYear = year[index]

    seasonIndex = calcSeasonModified(currentSeason)



if seasonIndex < 3:
    dailyMins5[seasonIndex] = list(dailyMins5[seasonIndex] + tempMin)
    dailyMeans5[seasonIndex] = list(dailyMeans5[seasonIndex] + tempMean[2])
    dailyMeans4[seasonIndex] = list(dailyMeans4[seasonIndex] + tempMean[1])
    dailyMeans3[seasonIndex] = list(dailyMeans3[seasonIndex] + tempMean[0])
    dailyMaxs5[seasonIndex] = list(dailyMaxs5[seasonIndex] + tempMax)
    dailyMortality[seasonIndex] = list(dailyMortality[seasonIndex] + tempMortalityList)
    dailyWeekday[seasonIndex] = list(dailyWeekday[seasonIndex] + tempWeekday)
    dailyUnsmthMort[seasonIndex] = list(dailyUnsmthMort[seasonIndex] + tempUnsmthMort)

# clear temp lists
tempMin = []
tempMean = [[], [], []]
tempMax = []
tempMortalityList = []
tempWeekday = []
tempUnsmthMort = []
"""
