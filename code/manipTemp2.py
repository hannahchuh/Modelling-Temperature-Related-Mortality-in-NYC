import pickle
import numpy as np
from scipy import stats
import re
import datetime
from decimal import Decimal
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv

def sameSeason( pMonth, cMonth ):
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

def calcSeason( monthNum ):
    if monthNum == 12 or monthNum == 1 or monthNum == 2:
        return 0
    if monthNum == 3 or monthNum == 4 or monthNum == 5:
        return 1
    if monthNum == 6 or monthNum ==7 or monthNum == 8:
        return 2
    if monthNum == 9 or monthNum == 10 or monthNum == 11:
        return 3

def calcListAvg( dataSet ):
    sum = 0
    for i in range(len(dataSet)):
        sum += dataSet[i]
    return Decimal(sum)/Decimal(len(dataSet))

# initialize variables
dailyMins = [[],[],[],[]]
dailyMaxs =[[],[],[],[]]
dailyMeans =[[],[],[],[]]
absMins = [[],[],[],[]]
absMaxs = [[],[],[],[]]
tempMinList = []
tempMeanList = []
tempMaxList = []

# load data
tempDict = pickle.load(open("longCompiledNY.csv", 'rb'))
year = tempDict['year']
month = tempDict['month']
day = tempDict['day']
minTemps = tempDict['minTemp']
meanTemps = tempDict['meanTemp']
maxTemps = tempDict['maxTemp']

#find 1st season of 1st yr and last season of last yr
for i in range(len(day)):
    if year[i] == 1973 and (month[i] == 1 or month[i] == 2):
        startIndex = i + 1
    if year[i] == 2015 and month[i] == 12:
        endIndex = i - 1
        break
# start and endIndex checked

# select part of lists
year = year[startIndex:endIndex+1]
month = month[startIndex:endIndex+1]
day = day[startIndex:endIndex+1]
minTemps = minTemps[startIndex:endIndex+1]
meanTemps = meanTemps[startIndex:endIndex+1]
maxTemps = maxTemps[startIndex:endIndex+1]

#initalize counters/trackers
index = 0
currentSeason = currentMonth = 0

#find seasonal daily min/max/mean and abs min/max
while index < len(year):
    #update previousMonth
    currentSeason = month[index]
    currentMonth = month[index]

    #iterate through a season
    while(sameSeason(currentSeason, currentMonth)) and index < len(year):
        currentMonth = month[index]

        #add to temp lists
        tempMinList.append(minTemps[index])
        tempMeanList.append(meanTemps[index])
        tempMaxList.append(maxTemps[index])

        #update index and previousMonth
        index += 1
        if index <len(year):
            currentMonth = month[index]

    seasonIndex = calcSeason(currentSeason)
    absMins[seasonIndex].append(min(tempMinList))
    absMaxs[seasonIndex].append(max(tempMaxList))
    dailyMins[seasonIndex].append(Decimal(sum(tempMinList))/Decimal(len(tempMinList)))
    dailyMeans[seasonIndex].append(Decimal(sum(tempMeanList))/Decimal(len(tempMeanList)))
    dailyMaxs[seasonIndex].append(Decimal(sum(tempMaxList))/Decimal(len(tempMaxList)))

    #clear temp lists
    tempMinList = []
    tempMeanList = []
    tempMaxList = []

'''
NOTES:
    absMins, absMaxs, dailyMins5, dailyMeans, dailyMaxs5 are correct lengths (42 for winter(index0) and 43 for rest)
    ALL lists have been checked (see "Checkings" doc)
'''
sampleDict = {}

dailyMinsCopy = list(dailyMins[0])

dailyMinsCopy[0] = Decimal('0')
sampleDict.update({'Decimal':dailyMinsCopy})

#cast all numbers to float64s
for i in range(4):
    for j in range(len(dailyMins[i])):
        dailyMins[i][j] = np.float64(dailyMins[i][j])
        dailyMeans[i][j] = np.float64(dailyMeans[i][j])
        dailyMaxs[i][j] = np.float64(dailyMaxs[i][j])
        absMins[i][j] = np.float64(absMins[i][j])
        absMaxs[i][j] = np.float64(absMaxs[i][j])

#testing
for i in range(4):
    print len(dailyMins[i])
    print len(dailyMeans[i])

#teseting


sampleDict.update({'Float':dailyMins[0]})

for keys, values in sampleDict.iteritems():
    print values

dictLength = len(sampleDict)
tempList = []
with open("compareDecimalFloat.csv", "wb") as fileObj:
    fileWriter = csv.writer(fileObj)
    listLength = len(sampleDict.itervalues().next())
    for index in range(listLength):
        for key in sampleDict:
            tempList.append( sampleDict[key][index])
        fileWriter.writerow(tempList)
        tempList = []

#axis
axis = np.arange(1,44)
slope, intercept, r_value, p_value, std_err = stats.linregress(axis, absMins[3])
print slope
print "R VALUE"
print r_value
print r_value*r_value
"""
line = slope*axis+intercept
xAxis = np.linspace(1973,2015,43)
plt.plot(xAxis, absMaxs[1])
plt.plot(xAxis, line)
plt.show()
"""

#plot graphs

#add 'nan' to the DJF in order to graph
#dailyMins5[0].insert(0,Decimal('nan'))
#dailyMeans[0].insert(0,Decimal('nan'))
#   dailyMaxs5[0].insert(0,Decimal('nan'))
absMaxs[0].insert(0,Decimal('nan'))
absMins[0].insert(0,Decimal('nan'))

xAxis = np.linspace(1973, 2015, 43)
fig = plt.figure(figsize=(11,6))
ax = fig.add_subplot(111)
ax.set_position([0.1,0.1,0.67,0.8])
fig.suptitle('absMax')                               #change

ax.plot(xAxis, absMaxs[1], label="winter")            #change
ax.plot(xAxis, absMins[1], label="spring")            #change
ax.plot(xAxis, dailyMins[1], label="summer")            #change
ax.plot(xAxis, dailyMeans[1], label="fall")              #change
leg= ax.legend(bbox_to_anchor=(1.05, 1), loc=2)

plt.ylabel('temperature')
plt.xlabel('year')
plt.ylim([-10,120])
plt.show()
#fig.savefig("absMaxs")                                #change
