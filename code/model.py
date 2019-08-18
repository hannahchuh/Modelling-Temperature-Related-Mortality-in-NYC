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
smoothedTemp = []

# smooth temperature set
smoothedMortality = rollingAvg(30, mortality)
smoothedTemp = rollingAvg(5, meanTemps)                               # change this as desired

# create subSmoothMort list
for i in range(len(smoothedMortality)):
    if smoothedMortality[i] == Decimal('nan'):
        subSmoothMortality.append(Decimal('nan'))
    else:
        subSmoothMortality.append(Decimal(mortality[i] - smoothedMortality[i]))

#print subSmoothMort                                            # DELETE
#print smoothedTemp                                                  # DELETE
#print smoothMort                                             # DELETE

# calc lowerVal for 80% before casting to floats
lowerVal = calcPercentile(Decimal('0.9'), smoothedTemp[4:])

# cast as floats
for i in range(len(smoothedMortality)):
    smoothedTemp[i] = np.float64(smoothedTemp[i])
    subSmoothMortality[i] = np.float64(subSmoothMortality[i])
    meanTemps[i] = np.float64(meanTemps[i])
    minTemps[i] = np.float64(minTemps[i])
    maxTemps[i] = np.float64(maxTemps[i])
    smoothedMortality[i] = np.float64(smoothedMortality[i])          # DELETE

#print smoothedTemp                                                  # DELETE
#print smoothMort                                             # DELETE


#VARIOUS ORDER FIT

# fit line
popt, pcov = opt.curve_fit(fourthOrder, smoothedTemp[29:], subSmoothMortality[29:])
print popt
line = []
count = 0
for i in range(29, len(smoothedTemp)):
    tempVal = smoothedTemp[i]
    #yVal = popt[0] + popt[1]*tempVal + popt[2]*tempVal*tempVal + popt[3]*tempVal*tempVal*tempVal + popt[4]*tempVal*tempVal*tempVal*tempVal
    #if
    yVal = fourthOrder(smoothedTemp[i], popt[0], popt[1], popt[2], popt[3], popt[4])
    line.append(yVal)
    count += 1

print "count is: " + str(count)
plt.plot(smoothedTemp[29:], line, 'ro')
plt.scatter(smoothedTemp[29:], subSmoothMortality[29:])
plt.ylabel('Mortality anomaly', fontsize=15)
plt.xlabel('Temperature ($^\circ$F)', fontsize = 15)
plt.title("Fourth order polynomial", fontsize=15)
plt.show()


'''
#LINEAR FIT WITH 80% AND UP
# x and y axis
xLine = []
yLine = []

# plot line and scatter plot
print lowerVal
for i in range(29, len(subSmoothMort)):
    if smoothedTemp[i] > lowerVal:
        xLine.append(smoothedTemp[i])
        yLine.append(subSmoothMort[i])
print xLine
print yLine

#plt.scatter(smoothedTemp, subSmoothMort)
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
# PRINT LINEAR FIT FOR smoothedTemp AND subSmoothMort WITH LINREGRESS

slope, intercept, r_value, p_value, std_err = stats.linregress(smoothedTemp, subSmoothMort)
print "SLOPE IS: " + str(slope)
line = []
for i in range(29, len(smoothedTemp)):
    line.append(np.float64(slope*smoothedTemp[i])+intercept)
plt.plot(smoothedTemp, line)
'''

'''
# SCIKIT LINEAR REGRESSION - ALL SEASONS
regr = linear_model.LinearRegression()
regr.fit((np.transpose(smoothedTemp[29:])).reshape(5085,1), (np.transpose(subSmoothMort[29:])).reshape(5085,1))
print regr.coef_
print regr.intercept_

line = []
for i in range(29, len(smoothedTemp)):
    line.append(np.float64(regr.coef_*smoothedTemp[i])+regr.intercept_)
plt.plot(smoothedTemp[29:], line)
plt.scatter(smoothedTemp[29:], subSmoothMort[29:])
plt.show()
'''

# separate out summer and winter

#initialize
startIndex = 0
endIndex = 0
dailyMins = [[],[]]                     # 0th index is winter
dailyMeans = [[],[]]
dailyMaxs = [[],[]]
dailyMortality = [[],[]]
dailyUnsmoothedMort = [[],[]]
dailyWeekday = [[],[]]
tempMinList = []
tempMeanList = []
tempMaxList = []
tempMortalityList = []
tempUnsmoothedList = []
tempDayList = []
winterCoeff = [[],[]]                   # 0th index is slope, 1st index is intercept
summerCoeff = [[],[]]
fitStart = 0
fitEnd = 0
predictStart = 0
predictEnd = 0
predictTemp = []
predictWeekday= []
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
        tempMinList.append(minTemps[index])
        tempMeanList.append(meanTemps[index])
        tempMaxList.append(maxTemps[index])
        tempMortalityList.append(subSmoothMortality[index])
        tempDayList.append(weekday[index])
        tempUnsmoothedList.append(mortality[index])

        #update index and previousMonth
        index += 1
        if index <len(year):
            currentMonth = month[index]

    seasonIndex = calcSeasonModified(currentSeason)
    if seasonIndex < 3:
        dailyMins[seasonIndex] = list( dailyMins[seasonIndex] + tempMinList )
        dailyMeans[seasonIndex] = list( dailyMeans[seasonIndex] + tempMeanList )
        dailyMaxs[seasonIndex] = list( dailyMaxs[seasonIndex] + tempMaxList )
        dailyMortality[seasonIndex] = list( dailyMortality[seasonIndex] + tempMortalityList)
        dailyWeekday[seasonIndex] = list( dailyWeekday[seasonIndex] + tempDayList)
        dailyUnsmoothedMort[seasonIndex] = list(dailyUnsmoothedMort[seasonIndex] + tempUnsmoothedList)

    #clear temp lists
    tempMinList = []
    tempMeanList = []
    tempMaxList = []
    tempMortalityList = []
    tempDayList = []
    tempUnsmoothedList = []

#len of dailyMins5/Means/Maxs' list is correct (JJA - 14, DJF - 13)          # RECHECK THIS

"""
# scikit linear regression - cycle through with 80/20 for winter
tempList = dailyMeans                                                       # change as desired
                                                                            # CHANGE THIS SO INDECES ARE NOT HARD CODED
for i in range(235):                                #80% can be shifted up 234 times
    regr = linear_model.LinearRegression()

    #fit
    fitStart = i
    fitEnd = 1173 - 234 + i
    regr.fit((np.transpose(tempList[0][fitStart:fitEnd+1])).reshape(940,1), (np.transpose(dailyMortality[0][fitStart:fitEnd+1])).reshape(940,1))

    winterCoeff[0].append(regr.coef_)
    winterCoeff[1].append(regr.intercept_)

    #predict
    predictStart = fitEnd + 1
    predictEnd = 1173

    predictTemp = tempList[0][predictStart:predictEnd+1]
    predictStart = 0
    predictEnd = i - 1

    predictTemp = list(predictTemp + tempList[0][predictStart:predictEnd+1])
    plt.plot(predictTemp, regr.predict((np.transpose(predictTemp)).reshape(234,1)), 'ro')
    plt.scatter(tempList[0], dailyMortality[0])
    #plt.show()

wSlopeAvg = np.float64(sum(winterCoeff[0]))/len(winterCoeff[0])
wIntercepAvg = np.float64(sum(winterCoeff[1]))/len(winterCoeff[1])

print wSlopeAvg
print wIntercepAvg

# scikit linear regression - cycle through with 80/20 for summer
print len(dailyMortality[1])
print len(dailyMortality[0])

tempList = dailyMeans
for i in range(258):                                #80% can be shifted up 234 times
    regr = linear_model.LinearRegression()

    #fit
    fitStart = i
    fitEnd = 1029 - 257 + i

    regr.fit((np.transpose(tempList[1][fitStart:fitEnd+1])).reshape(773,1), (np.transpose(dailyMortality[1][fitStart:fitEnd+1])).reshape(773,1))

    summerCoeff[0].append(regr.coef_)
    summerCoeff[1].append(regr.intercept_)

    #predict
    predictStart = fitEnd + 1
    predictEnd = 1029

    predictTemp = tempList[0][predictStart:predictEnd+1]
    predictStart = 0
    predictEnd = i - 1

    predictTemp = list(predictTemp + tempList[1][predictStart:predictEnd+1])
    plt.plot(predictTemp, regr.predict((np.transpose(predictTemp)).reshape(257,1)), 'ro')
    plt.scatter(tempList[1], dailyMortality[1])
    #plt.show()

sSlopeAvg = np.float64(sum(summerCoeff[0]))/len(summerCoeff[0])
sIntercepAvg = np.float64(sum(summerCoeff[1]))/len(summerCoeff[1])

print sSlopeAvg,
print " "
print sIntercepAvg
"""

# scikit lin regrss w/ multiple x values

# scikit linear regression - cycle through with 80/20 for winter
tempList = dailyMeans                                                       # change as desired
                                                                            # CHANGE THIS SO INDECES ARE NOT HARD CODED
xMatrix = [[],[]]
for i in range(235):                                #80% can be shifted up 234 times
    regr = linear_model.LinearRegression()

    #fit
    fitStart = i
    fitEnd = 1173 - 234 + i
    xMatrix[0] = tempList[0][fitStart:fitEnd+1]
    xMatrix[1] = dailyWeekday[0][fitStart:fitEnd+1]
    regr.fit((np.transpose(xMatrix)).reshape(940,2), (np.transpose(dailyMortality[0][fitStart:fitEnd+1])).reshape(940,1))

    winterCoeff[0].append(regr.coef_)
    winterCoeff[1].append(regr.intercept_)

    #predict
    predictStart = fitEnd + 1
    predictEnd = 1173

    predictTemp = tempList[0][predictStart:predictEnd+1]
    predictWeekday = dailyWeekday[0][predictStart:predictEnd+1]
    predictStart = 0
    predictEnd = i - 1

    """
    predictTemp = list(predictTemp + tempList[0][predictStart:predictEnd+1])
    predictWeekday = list(predictWeekday + dailyWeekday[0][predictStart:predictEnd+1]
    xMatrix[0] = predi
    xMatrix[1] = predictWeekday
    plt.plot(predictTemp, regr.predict((np.transpose(xMatrix)).reshape(234,2)), 'ro')
    plt.scatter(tempList[0], dailyMortality[0])
    #plt.show()
    """

wSlopeAvg = np.float64(sum(winterCoeff[0]))/len(winterCoeff[0])
wIntercepAvg = np.float64(sum(winterCoeff[1]))/len(winterCoeff[1])

print wSlopeAvg
print wIntercepAvg

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
    """
    #predict
    predictStart = fitEnd + 1
    predictEnd = 1029

    predictTemp = tempList[0][predictStart:predictEnd+1]
    predictStart = 0
    predictEnd = i - 1

    predictTemp = list(predictTemp + tempList[1][predictStart:predictEnd+1])
    plt.plot(predictTemp, regr.predict((np.transpose(predictTemp)).reshape(257,1)), 'ro')
    plt.scatter(tempList[1], dailyMortality[1])
    #plt.show()
    """

sSlopeAvg = np.float64(sum(summerCoeff[0]))/len(summerCoeff[0])
sIntercepAvg = np.float64(sum(summerCoeff[1]))/len(summerCoeff[1])

print sSlopeAvg,
print " "
print sIntercepAvg

#"""
# PRINT DAY OF WEEK VS MORTALITY BAR GRAPH

#day of week vs mortality
for i in range(2):
    for j in range(len(dailyMortality[i])):                                # change this to the dailiymortality for diff graph
        weekDayVal = dailyWeekday[i][j]
        print weekDayVal
        weekDayMortality[i][weekDayVal-1].append(dailyMortality[i][j])     # same here

for i in range(2):
    for j in range(len(weekDayMortality[0])):
        weekDayMortality[i][j] = np.float64(sum(weekDayMortality[i][j]))/len(weekDayMortality[i][j])
    print weekDayMortality[i]

    xAxisLabels = []
    xAxisLabels.append('Sunday')
    xAxisLabels.append('Monday')
    xAxisLabels.append('Tuesday')
    xAxisLabels.append('Wednesday')
    xAxisLabels.append('Thursday')
    xAxisLabels.append('Friday')
    xAxisLabels.append('Saturday')
    xAxis = np.arange(7)
    plt.clf()
    plt.xticks(xAxis, xAxisLabels)
    plt.bar(xAxis, weekDayMortality[i])

    # titles
    if i == 0:
        plt.title("Winter", fontsize=15)
    else:
        plt.title("Summer", fontsize=15)
    plt.ylabel("Mortality anomaly", fontsize=15)
    plt.xlabel("Weekday", fontsize=15)

    plt.show()
#"""

"""
# SCIKIT LINEAR REGRESSION WITH WINTER/SUMMER UNSMOOTHED TEMP
regr = linear_model.LinearRegression()
regr.fit((np.transpose(dailyMeans[1][:1030])).reshape(1030,1), (np.transpose(dailyMortality[1][:1030])).reshape(1030,1))
print regr.coef_
print regr.intercept_

#calc line
line = []
for i in range(len(dailyMeans[1])):
    line.append(np.float64(regr.coef_*dailyMeans[1][i])+regr.intercept_)

plt.plot(dailyMeans[1][1030:], regr.predict((np.transpose(dailyMeans[1][1030:])).reshape(258,1)), 'ro')

#plot scatters, line
plt.plot(dailyMeans[1], line)
plt.scatter(dailyMeans[1], dailyMortality[1])
plt.show()
"""

