"""
Read in date and mortality data from 1987-2000. Read in date and temp data for
1973-2015. Find dailymin/max/mean temps. Combine all data and write to file
(both pickled and not pickled)
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
shortSet = {}
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

# READ MORTAlITY DATA
nyFile = open( 'mortalityNY.csv' )
csv_ny = csv.reader( nyFile )

# skip first row of NY CITY CSV file
csv_ny.next()

for index, row in enumerate( csv_ny ):      #index for testing purposes, has no real purpose

    # read in date
    dateString = row[ 2 ]
    dateObj = datetime.datetime.strptime( dateString, "%Y%m%d" ).date()
    shortAllDates.append( dateObj )
    shortYear.append( dateObj.year )
    shortMonth.append( dateObj.month )
    shortDay.append( dateObj.day )

    # read in num of deaths
    deaths.append( int( row[ 8 ] ) )

    # read in weekday
    shortWeekDay.append( int( row[ 3 ]))

# add date and mortality to dict
shortSet.update( {'day':shortDay} )
shortSet.update( {'month':shortMonth} )
shortSet.update( {'year':shortYear} )
shortSet.update( {'mortality':deaths} )
shortSet.update( {'weekday':shortWeekDay})

# tested correct

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

# loop through file
for line in climateFile:
    thisLine = line.rsplit()
    dateStr = thisLine[2][:8]
    dateObj = datetime.datetime.strptime(dateStr, "%Y%m%d").date()

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

    # check for missing temp readings
    if not("*" in tempStr):
        dayTemps.append( Decimal (tempStr))

    # check for missing dew point
    if not("*" in tempDewPt):
        dayDewPts.append( Decimal (tempDewPt))

# add last day
longAllDates.append(previousDateObj)
longYear.append(previousDateObj.year)
longMonth.append(previousDateObj.month)
longDay.append(previousDateObj.day)
minTemps.append(min(dayTemps))
maxTemps.append(max(dayTemps))
meanTemps.append(Decimal(sum(dayTemps)) / Decimal(len(dayTemps)))
minDewPts.append(min(dayDewPts))
meanDewPts.append(Decimal(sum(dayDewPts))/Decimal(len(dayDewPts)))
maxDewPts.append(max(dayDewPts))

'''NOTES:
    no nans in the temp lists
    lengths of all lists in longSet = 15705 (this is correct if 2000 is counted as leap year)
'''

# add temperatures to longSet
longSet.update( { 'month':longMonth } )
longSet.update( { 'year':longYear } )
longSet.update( { 'day':longDay } )
longSet.update( { 'minTemp':minTemps } )
longSet.update( { 'maxTemp':maxTemps} )
longSet.update( { 'meanTemp':meanTemps } )
longSet.update( { 'minDewPoint':minDewPts } )
longSet.update( { 'meanDewPoint':meanDewPts } )
longSet.update( { 'maxDewPoint':maxDewPts } )

# find shortSet interval in longSet temps
for i in range(len(minTemps)):
    if(longYear[i] == 1987 and longMonth[i] == 1 and longDay[i] == 1):
        startIndex = i

    if(longYear[i] == 2000 and longMonth[i] == 12 and longDay[i] == 31):
        endIndex = i
        break;

# create shortTemps lists
shortMinTemps = minTemps[startIndex:endIndex+1]
shortMeanTemps = meanTemps[startIndex:endIndex+1]
shortMaxTemps = maxTemps[startIndex:endIndex+1]
shortMinDewPts = minDewPts[startIndex:endIndex+1]
shortMeanDewPts = meanDewPts[startIndex:endIndex+1]
shortMaxDewPts = maxDewPts[startIndex:endIndex+1]

# add shortTemps to shortSet
shortSet.update( { 'minTemp':shortMinTemps } )
shortSet.update( { 'meanTemp':shortMeanTemps } )
shortSet.update( { 'maxTemp':shortMaxTemps } )
shortSet.update( { 'minDewPt':shortMinDewPts } )
shortSet.update( { 'meanDewPt':shortMeanDewPts } )
shortSet.update( { 'maxDewPt':shortMaxDewPts } )

for keys, values in shortSet.iteritems():
    print len(values)

'''
NOTES:
    all lenghts of shorTemps = 5114
    shortSet correct - i would check dewpoint tho
'''

# pickling
with open('longCompiledNY.csv', 'wb') as handle:
    pickle.dump(longSet, handle)

with open('shortCompiledNY.csv', 'wb') as handleTwo:
    pickle.dump(shortSet, handleTwo)

'''
NOTES:
    both dicts tested as correct - i would check dewpoint tho
'''

# write longSet to csv file
sampleDict = longSet
tempList = []
dictLength = len(sampleDict)
with open("longCompiledNYReadable.csv", "wb") as fileObj:
    fileWriter = csv.writer(fileObj)
    listLength = len(sampleDict.itervalues().next())
    for index in range(listLength):
        for key in sampleDict:
            tempList.append( sampleDict[key][index])
        fileWriter.writerow(tempList)
        tempList = []

# write shortSet to csv file
sampleDict = shortSet
dictLength = len(sampleDict)
tempList = []
with open("shortCompiledNYReadable.csv", "wb") as fileObj:
    fileWriter = csv.writer(fileObj)
    listLength = len(sampleDict.itervalues().next())
    for index in range(listLength):
        for key in sampleDict:
            tempList.append( sampleDict[key][index])
        fileWriter.writerow(tempList)
        tempList = []

"""
for keys, values in shortSet.iteritems():
    print keys
"""
