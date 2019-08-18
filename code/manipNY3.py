# manipulate NY Data

###MODULES
import pickle
import matplotlib.pyplot as plt
import numpy as np
import csv
from decimal import Decimal
from decimal import ROUND_CEILING

###FUNCTIONS
def rollingAvg( lag, oldSet ):
    newSet = []

    # insert nan's
    for i in range(0, lag - 1):
        newSet.append(Decimal('nan'))

    for i in range((lag - 1), len(oldSet)):
        sum = 0
        for j in range(lag):
            sum += oldSet[i - j]

        avg = sum / Decimal(lag)
        newSet.append(Decimal(avg))

    return newSet

def calcPercentile(percent, set):
    if percent == Decimal('1.0'):
        return max(set)

    #convert percent to the appropiate index
    pValue = percent * len(set)

    set = sorted(set)

    sampleList = set
    listLength = len(sampleList)
    with open("test.csv", "wb") as fileObj:
        fileWriter = csv.writer(fileObj)
        for index in range(listLength):
            fileWriter.writerow([sampleList[index]])

    print "non-rounded approximate index is: " + str(pValue)

    #check if percent is an integer
    if pValue % 1 == 0:
        pValue = int(pValue)

        print "taking average of " + str(pValue - 1) + " and " + str(pValue) + " indices"

        #take average of values at indices percent and percent - 1
        return (set[pValue - 1] + set[pValue]) / Decimal('2')

    #if percentage needs to be rounded
    else:
        #round number up to nearest integer
        pValue = pValue.to_integral_exact(rounding=ROUND_CEILING)
        pValue = int(pValue)

        print "accessing at " + str(pValue-1) + " index"
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
    print "\tLower Index: " + str(lowerIndex) + "     Upper Index: " + str(index-1)

def priorAnomalies( index, threshold, dataSet ):
    listLength = len(dataSet)

    if( index < 0 or index >= len( dataSet)):
        return 0

    for i in range(1,5):
        if(index-i > 0 and not(np.isnan(float( dataSet[i]))) and dataSet[index-i] >= threshold):
            return 0

    return 1

def isLeapDay(year, month, day):
    if year == 1988 or year == 1992 or year == 1996 or year == 2000:
        if month == 2:
            if day == 29:
                return True

    return False

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
            if(month[i] ==1 and day[i]==1):
                print mortality[i]
            avgDayArr[dayCount] += Decimal(mortality[i])
            if(dayCount == 0):
                print "\t" + str(year[i]) + " " + str(month[i]) + " " + str(day[i]) + "|" + str(avgDayArr[dayCount])
            dayCount += 1

    print "final count: " + str(avgDayArr[0])
    for i in range(365):
        avgDayArr[i] = avgDayArr[i]/Decimal(14)

    leapDayAvg = leapDayAvg/Decimal(4)
    return leapDayAvg

def findPercentileRange( lowerPercent, upperPercent, tempSet, mortalitySet ):
    if (lowerPercent <= 0 or upperPercent <= 0):
        print "INVALID PERCENT! -- LESS THAN 0"
        return

    startingIndex = len(tempSet) - len(mortalitySet)
    print str(startingIndex) + " is starting index"
    rangeArr = []
    anomDays = []
    preDays = [[],[],[],[]]
    postDays = [[],[],[],[]]
    numAnomDays = 0

    listLength = len(tempSet)
    lowerVal = calcPercentile(lowerPercent, tempSet)
    upperVal = calcPercentile(upperPercent, tempSet)

    #testing
    print "Lower val: " + str(lowerVal)
    print "Upper val: " + str(upperVal)
    #endtesting

    for i in range(startingIndex, listLength):
        if(     ( not np.isnan(float(tempSet[i])) )
            and ( not np.isnan(float(mortalitySet[i-startingIndex])))
            and tempSet[i] <= upperVal
            and tempSet[i] >= lowerVal
          ):
            numAnomDays += 1
            anomDays.append(mortalitySet[i-startingIndex])
            for j in range(1, 5):

                #days following
                if( i + j < listLength ):
                    postDays[j-1].append(Decimal(mortalitySet[i+j-startingIndex]))

                #days preceding
                if( i - j >= 0 and not np.isnan(float(mortalitySet[i-startingIndex-j]))):
                    preDays[j-1].append(Decimal(mortalitySet[i-j-startingIndex]))

    anomDaysAvg = sum(anomDays)/Decimal(numAnomDays)

    print "ANOM DAY AVERAGE: " + str(anomDaysAvg)

    for i in range(4):
        preDays[i] = sum(preDays[i])/Decimal(len(preDays[i]))
        rangeArr.append(preDays[i])

    rangeArr.append(anomDaysAvg)

    for i in range(4):
        postDays[i] = sum(postDays[i])/Decimal(len(postDays[i]))
        rangeArr.append(postDays[i])

    return rangeArr

######################################################################

###"MAIN"
nyDict = pickle.load(open("compiledNYData.csv", 'rb'))

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

# smoothing
smoothedMortality = rollingAvg(M_LAG, mortality)
smoothedMin = rollingAvg(3, minTemps)
smoothedMax = rollingAvg(3, maxTemps)
smoothedMean = rollingAvg(3, meanTemps)                                 #tested smooth are correct

#fill subSmoothMort (len 5085)
for index in range(M_LAG-1, len(smoothedMortality)):
    subSmoothMortality.append(mortality[index] - smoothedMortality[index]) #tested subSmooth are correct

#fill subMeanMortality
meanMortality = sum(mortality)/Decimal(len(mortality))
print "mean mortality is: " + str(meanMortality)                    #mean mortality is correct
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


print subDayMortality
print subSmoothMortality
print subMeanMortality

# graphing testing
selectedList = subDayMortality
rangeOne = findPercentileRange(Decimal('0.95'), Decimal('1.00'), minTemps, selectedList)
rangeTwo = findPercentileRange(Decimal('0.50'), Decimal('0.55'), minTemps, selectedList)
rangeThree = findPercentileRange(Decimal('0.05'), Decimal('0.10'), minTemps, selectedList)
xAxis = np.linspace(-4,4,9)
plt.plot(xAxis, rangeOne)
plt.show()

'''
#SHOW GRAPHS OF ALL THREE ALTERED MORTALITY SETS
xAxis = np.linspace(1987, 2000, 5114)
#plt.plot(xAxis, subMeanMortality)
plt.plot(xAxis, subDayMortality)
xAxis = np.linspace(1987, 2000, 5085)
#plt.plot(xAxis, subSmoothMort)
plt.show()
'''












'''
# lag function
# for mortality do lag of about 30

# look at what happens after hot days
# smoothe the temperature w/ maybe lag = 3 (try different temp lists and lags - 10?)
# pick a temperature threshold, or look for 95% (try different percentiles)
# probably below 80% will show nothing (maybe even 90%)
# compare it to midrange (look at average)
# look through the list and find hot days
# look at mortality and see what happens right after hot days
#average all the mortality in the following four days

#put 50-55, and then 90-95, greater than 95, 85-90, 80-85
#do this to all the min, mean, max sets (3 graphs with 5 lines)
#probably going to find that there is a seasonal cycle that we want to remove

#since we want to be looking at the anomalies
#one way to remove it is to generate a very smoothed version of mortality anomalies
#at each point remove the smoothed value
'''
