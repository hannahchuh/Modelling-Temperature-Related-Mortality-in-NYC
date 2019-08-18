import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import math

currentDirPath = os.getcwd()
subDir = os.path.join(currentDirPath, "subWinterSummer\\")
subDir = os.listdir(subDir)

subDir = subDir[:-1]            # remove "readable" diirectory

tempDict = []
futMort = []
pastMort = []

for file in subDir:
    filePath = 'C:/Users/h/PycharmProjects/untitled2/subWinterSummer/' + file
    tempDict = pickle.load(open(filePath))
    futMort.append(tempDict['futSubWinterSummer'])
    pastMort.append(tempDict['pastSubWinterSummer'])


pastMedians = []
futMedians = []
pastYearAvg =[]
futYearAvg = []
avgPastMort = []
avgFutMort = []
lowerFutMortBound = []
lowerPastMortBound = []
higherFutMortBound = []
higherPastMortBound = []

futBoxPlot = []
pastBoxPlot = []

print "LEN: "
print len(futMort[0])
# for every day
for i in range(len(futMort[0])):
    # for every model
    for j in range(len(futMort)):
        if not math.isnan(futMort[j][i]):
            futYearAvg.append(futMort[j][i])        # list of all mortality proj from the models
    futYearAvg.sort()
    lowerFutMortBound.append(futYearAvg[1])
    higherFutMortBound.append(futYearAvg[len(futYearAvg)-2])
    avgFutMort.append(sum(futYearAvg)/np.float64(len(futYearAvg)))
    futBoxPlot.append(futYearAvg)
    futMedians.append(futYearAvg[len(futYearAvg)/2])
    futYearAvg = []

for i in range(len(pastMort[0])):
    for j in range(len(pastMort)):
        if not math.isnan(pastMort[j][i]):
            pastYearAvg.append(pastMort[j][i])
    pastYearAvg.sort()
    lowerPastMortBound.append(pastYearAvg[1])
    higherPastMortBound.append(pastYearAvg[len(pastYearAvg) - 2])
    avgPastMort.append(sum(pastYearAvg) / np.float64(len(pastYearAvg)))
    pastBoxPlot.append(pastYearAvg)
    pastMedians.append(pastYearAvg[len(pastYearAvg)/2])
    pastYearAvg = []


# appended together into list approach
fillerList = []
for i in range(20):
    fillerList.append(0)
pastFutList = pastBoxPlot + fillerList + futBoxPlot
xScale = []
for i in range(1988,2081):
    if i%4 == 0:
        xScale.append(str(i))
    else:
        xScale.append("")
xScale[0] = "1988"
fig = plt.figure(figsize=(20,15))
boxPlot = plt.boxplot(pastFutList, whis=[10,90])
futTicks = plt.xticks(np.arange(1,94), xScale, rotation=45)
plt.title("Mortality projections across seven CMIP5 models", fontsize = 15)
plt.ylabel("Total mortality anomaly", fontsize = 15)
plt.xlabel("Year", fontsize = 15)
#plt.tick_params(axis='x', pad=4)
plt.show()


"""
xScale = np.arange(59)
plt.hold(True)
xAxis = np.arange(2021, 2081)
#futBox = plt.boxplot(futBoxPlot, usermedians = futMedians, whis = [10,90])
futBox = plt.boxplot(futBoxPlot, whis=[10,90], meanline = True)
futTicks = plt.xticks(xScale, xAxis, rotation = 45)
plt.show()
plt.clf()

xScale = np.arange(12)
xAxis = np.arange(1988,2001)
#pastBox = plt.boxplot(pastBoxPlot, usermedians = pastMedians, whis = [10,90])
pastBox = plt.boxplot(pastBoxPlot, whis=[10,90], meanline = True)
pastTicks = plt.xticks(xScale, xAxis, rotation = 45)
plt.show()
#"""

"""
# plot everything as a scatter plot
xAxis = np.arange(2021,2081)
plt.hold(True)
plt.scatter(xAxis, avgFutMort)
plt.scatter(xAxis, lowerFutMortBound, color='green')
plt.scatter(xAxis, higherFutMortBound, color='green')

xAxis = np.arange(1988,2001)
plt.scatter(xAxis, avgPastMort)
plt.scatter(xAxis, lowerPastMortBound, color='green')
plt.scatter(xAxis, higherPastMortBound, color='green')
plt.show()
#"""

