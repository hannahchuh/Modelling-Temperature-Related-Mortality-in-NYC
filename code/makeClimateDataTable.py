"""
Read in date and climate data for 1987-2000. Combine all data and write to excel file
"""

# MODULES
import datetime
import pickle
import matplotlib.pyplot as plt
import numpy
from decimal import Decimal
import csv
import re

# FUNCTIONS
def rollingAvg(lag, oldSet):
    newSet = []

    # insert nan's
    for i in range(0, lag - 1):
        newSet.append(Decimal('nan'))

    for i in range((lag - 1), len( oldSet )):
        sum = 0
        for j in range(lag):
            sum += oldSet[i - j]

        avg = sum / Decimal(lag)
        newSet.append(Decimal(avg))

    return newSet

# VARIABLES (important ones)
longSet = {}
shortYear = []
shortMonth = []
shortDay = []
shortWeekDay = []
longYear = []
longMonth= []
longDay = []
deaths = []
minTemps = []
maxTemps = []
meanTemps = []
minDewPts = []
meanDewPts = []
maxDewPts = []
shortAllDates = []
longAllDates = []
dayTemps = []
dayDewPts = []

#output arrays
dates = []
temps = []
dewPts = []
times = []

'''
Notes:
    length of lists day, month, deaths, and year are all 5114 (this is correct)
'''

# READ TEMP DATA
climateFile = open("longTempNY.txt", 'r')

# skip first line (headers)
climateFile.readline()

# hardcode first date
previousDateObj = datetime.datetime.strptime("19730101", "%Y%m%d").date()
dates.append("19730101")
times.append("0000")

# loop through file
for line in climateFile:
    thisLine = line.rsplit()
    dateStr = thisLine[2][:8]
    dateObj = datetime.datetime.strptime(dateStr, "%Y%m%d").date()

    dates.append(dateStr)
    times.append(thisLine[2][8:])

    # if reached the end of one day's data set, calculate the min/mean/max
    # and add temperatures to the lists
    if previousDateObj != dateObj:
        longAllDates.append(previousDateObj)
        longYear.append( previousDateObj.year )
        longMonth.append( previousDateObj.month )
        longDay.append( previousDateObj.day )

        minDewPts.append( min( dayDewPts ) )
        maxDewPts.append( max ( dayDewPts ) )
        meanDewPts.append( Decimal( sum( dayDewPts ) )/Decimal( len( dayDewPts ) ) )

        minTemps.append( min( dayTemps ) )
        maxTemps.append( max( dayTemps ) )
        meanTemps.append( Decimal( sum( dayTemps ) )/Decimal( len ( dayTemps ) ) )

        dayTemps = []
        dayDewPts = []

    previousDateObj = dateObj

    # read in temperature
    tempStr = thisLine[21]
    tempDewPt = thisLine[22]

    temps.append(thisLine[21])
    dewPts.append(thisLine[22])

    # check for missing temp readings
    if not("*" in tempStr):
        dayTemps.append( Decimal (tempStr))

    # check for missing dew point
    if not("*" in tempDewPt):
        dayDewPts.append( Decimal (tempDewPt))

# add shortTemps to shortSet
longSet.update( { 'date':dates[:460020]} )
longSet.update( { 'time':times[:460020]} )
longSet.update( { 'temp':temps[:460020]} )
longSet.update( { 'dewPt':dewPts[:460020]} )

print len(dates)
print len(times)
print len(temps)
print len(dewPts)

for keys, values in longSet.iteritems():
    print len(values)

# write longSet to csv file
sampleDict = longSet
tempList = []
dictLength = len(sampleDict)
with open("cliamteDataTable.csv", "wb") as fileObj:
    fileWriter = csv.writer(fileObj)
    listLength = len(sampleDict.itervalues().next())
    for index in range(listLength):
        for key in sampleDict:
            tempList.append( sampleDict[key][index])
        fileWriter.writerow(tempList)
        tempList = []
