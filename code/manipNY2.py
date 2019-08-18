# manipulate NY Data

# ##MODULES
import pickle
import matplotlib.pyplot as plt
import numpy as np
import csv
from decimal import Decimal
from decimal import ROUND_CEILING

# ##FUNCTIONS
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

# fix this thing. make inclusive percentiles, find it at the lowest indice the desired percentile value occurs
def calcPercentile(percent, set):
    if percent == Decimal('1.0'):
        return max(set)

    # convert percent to the appropiate index
    pValue = percent * len(set)
    set = sorted(set)

    sampleList = set
    listLength = len(sampleList)
    with open("test.csv", "wb") as fileObj:
        fileWriter = csv.writer(fileObj)
        for index in range(listLength):
            fileWriter.writerow([sampleList[index]])

    print "non-rounded approximate index is: " + str(pValue)

    # check if percent is an integer
    if pValue % 1 == 0:
        pValue = int(pValue)

        print "taking average of " + str(pValue - 1) + " and " + str(pValue) + " indices"

        # take average of values at indices percent and percent - 1
        return (set[pValue - 1] + set[pValue]) / Decimal('2')

    # if percentage needs to be rounded
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




######################################################################

###"MAIN"
nyDict = pickle.load(open("compiledNYData.csv", 'rb'))

# setting up dicts and lists
deaths = nyDict['mortality']
minTemps = nyDict['minTemp']
maxTemps = nyDict['maxTemp']
meanTemps = nyDict['meanTemp']

# smoothing
smoothedMortality = rollingAvg(30, deaths)
smoothedMin = rollingAvg(3, minTemps)
smoothedMax = rollingAvg(3, maxTemps)
smoothedMean = rollingAvg(3, meanTemps)

#testingSTART
sampleDict = {}
sampleDict.update({'smoothMort':smoothedMortality})
sampleDict.update({'smoothedMin':smoothedMin})
sampleDict.update({'smoothedMax':smoothedMax})
sampleDict.update({'smoothedMean':smoothedMean})
for key,values in sampleDict.iteritems():
    print key
dictLength = len(sampleDict)
tempList = []
with open("smoothedDict.csv", "wb") as fileObj:
    fileWriter = csv.writer(fileObj)
    listLength = len(sampleDict.itervalues().next())
    for index in range(listLength):
        count = 0
        for key in sampleDict:
            count +=1
            tempList.append( sampleDict[key][index])
        fileWriter.writerow(tempList)
        tempList = []
#testingEND


# calculate lower percentiles
percent = Decimal('0.95')                                               # change percent as desired
minLowerPercentile = calcPercentile(percent, minTemps)
maxLowerPercentile = calcPercentile(percent, maxTemps)
meanLowerPercentile = calcPercentile(percent, meanTemps)


# select list + percentile to look at
lowerPercentile = minLowerPercentile                                    #change to min/max/mean as desired
selectedList = minTemps                                                 #change min/max/mean as desired

#initialize arrays
preDayArray1 = [[],[],[],[]]
postDayArray1 = [[],[],[],[]]
anomDay1 = 0
preDayArray2 = [[],[],[],[]]
postDayArray2 = [[],[],[],[]]
anomDay2 = 0

'''
#replace nans in selected list
print selectedList
for i in range(60):
    if np.isnan(float(selectedList[i])):
        print i
        selectedList[i] = Decimal('0')
'''


# look at days about temp threshold of 95% and mortality in days following + days before
numDays = 0
for i in range( len( selectedList ) ):

    if ( not(np.isnan( float( selectedList[ i ] ) ) ) ):
        #for temps above threshold & not a NaN
        if selectedList[ i ] >= lowerPercentile:
            #counters
            numDays+=1
            anomDay1 += Decimal( deaths[ i ] )

            print "95%: " + str( i )

            #print mortality for following/preceding days
            for j in range(1, 5):

                #check if index is in range of list
                if ( i + j ) < len( selectedList ):

                    #add days following
                    postDayArray1[ j - 1 ].append( Decimal( deaths[ i + j ] ) )

                #add days before
                if( ( i - (5- j) ) >  0 ):
                    preDayArray1[ j  - 1 ].append( Decimal( deaths[ i - (5-j) ] ) )

                    #print averageOfFour

anomDay1 = anomDay1/numDays
print "Number of days found: " + str( numDays )

'''
print "\t{0}, {1}, {2}, {3}".format(str(len(preDayArray1[0])), str(len(preDayArray1[1])), str(len(preDayArray1[2])),
                                    str(len(preDayArray1[3])))
print "\t{0}, {1}, {2}, {3}".format(str(len(postDayArray1[0])), str(len(postDayArray1[1])), str(len(postDayArray1[2])),
                                    str(len(postDayArray1[3])))
'''
'''
for i in range(15):
    print "---------"

#do the same thing for  50%

#change the percentile
percent = Decimal( '0.50' )                                             # change this value as desired
minPercentile = calcPercentile( percent, minTemps )

#calculate percentiles
percent = Decimal( '0.55' )
upperPercentile = calcPercentile( percent, minTemps )
lowerPercentile = minPercentile


print str(lowerPercentile) + " " + str(upperPercentile)

#find 50% days in selectedList
numDays = 0
for i in range( len( selectedList ) ):

    #for temps above threshold & not a NaN
    if not( np.isnan( float( selectedList[ i ] ) ) ) and selectedList[ i ] > lowerPercentile and selectedList[i] < upperPercentile:
        print "50-55%: " + str( i )

        #counters
        numDays+=1
        daysCounter = 0
        averageOfFour = 0
        anomDay2 += Decimal( selectedList[ i ] )

        # print mortality for following days
        for j in range(1,5):
            if ( i + j  ) < len(selectedList):
                daysCounter += 1
                averageOfFour += Decimal(deaths[i + j])
                postDayArray2[ j -  1 ].append(Decimal(deaths[i + j ]))

            if ( (i - j ) > 0):
                preDayArray2[ j - 1 ].append(Decimal(deaths[i - j]))

        averageOfFour = averageOfFour / Decimal(daysCounter)

anomDay2 = anomDay2/numDays
print "Number of days found: " + str( numDays )
'''
'''
print "\t{0}, {1}, {2}, {3}".format(str(len(preDayArray2[0])), str(len(preDayArray2[1])), str(len(preDayArray2[2])),
                                    str(len(preDayArray2[3])))
print "\t{0}, {1}, {2}, {3}".format(str(len(postDayArray2[0])), str(len(postDayArray2[1])), str(len(postDayArray2[2])),
                                    str(len(postDayArray2[3])))
'''

preDayAvg1 = []
preDayAvg2 = []
postDayAvg1 = []
postDayAvg2 = []

for index in range(4):
    preDayAvg1.append( sum( preDayArray1[ index ] )/Decimal( len( preDayArray1[ index ] ) ) )
    #preDayAvg2.append( sum ( preDayArray2[ index ] )/Decimal( len ( preDayArray2[ index ] ) ) )
    postDayAvg1.append( sum (postDayArray1[ index ] )/Decimal( len ( postDayArray1[ index ] ) ) )
    #postDayAvg2.append( sum (postDayArray2[ index ] )/Decimal( len ( postDayArray2[ index ] ) ) )



preDayAvg1.insert( 4, anomDay1)
#preDayAvg2.insert( 4, anomDay2)

averagedDays1 = preDayAvg1 + postDayAvg1
#averagedDays2 = preDayAvg2 + postDayAvg2

print len(averagedDays1)
print averagedDays1
#print len(averagedDays2)

xAxis = np.linspace(-4, 4, 9)
plt.plot( xAxis, averagedDays1 )
#plt.plot( xAxis, averagedDays2 )
plt.show()


print deaths[150]


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
