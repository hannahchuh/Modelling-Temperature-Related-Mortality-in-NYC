import os
import csv
import pickle
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt

# MAIN

# find path to bnu-esm historical
currentDirPath = os.getcwd()
histDir = os.path.join(currentDirPath, "histProj\\gfdl-esm")
histDir = os.listdir(histDir)

# initialize
month = []
year = []
day = []
meanTemps = []
dewPts = []
smoothMeans5 = []
histDict = {}
startIndex = 0
endIndex = 0

# read in temperature data
for file in histDir:
    filePath = 'C:/Users/h/PycharmProjects/untitled2/histProj/gfdl-esm/' + file
    monthData = io.loadmat(filePath)
    tempData = monthData[file[:-4]][0][2][0,0,:]
    print tempData
    plt.show()
    tempData = [np.float64(i) for i in tempData]
    meanTemps = list(meanTemps + list(tempData))
    for i in range(len(tempData)):
        day.append(int(i+1))
        month.append(int(file[12:14]))
        year.append(int(file[7:11]))

# add temperature data to dict
histDict.update({'month':month})
histDict.update({'year':year})
histDict.update({'meanTemps':meanTemps})
histDict.update({'day':day})
print len(year)

# read in dew point data
histDir = os.path.join(currentDirPath, "histDewProj\\gfdl-esm")
histDir = os.listdir(histDir)
for file in histDir:
    filePath = filePath = 'C:/Users/h/PycharmProjects/untitled2/histDewProj/gfdl-esm/' + file
    monthData = io.loadmat(filePath)
    tempData = monthData[file[:-4]][0][2][0, 0, :]
    tempData = [np.float64(i) for i in tempData]
    dewPts = list(dewPts + list(tempData))

histDict.update({'dewPts':dewPts})

xAxis = np.arange(len(histDict['dewPts']))
plt.plot(xAxis, histDict['dewPts'])
plt.show()


# export temperature data to csv (pickle)
with open('gfdl-esmHistCompiled.csv', 'wb') as handle:
    pickle.dump(histDict, handle)

# export temperature data to csv (readable)
sampleDict = histDict
tempList = []
dictLength = len(sampleDict)
with open("gfdl-esmHistCompiledReadable.csv", "wb") as fileObj:
    fileWriter = csv.writer(fileObj)
    listLength = len(sampleDict.itervalues().next())
    for index in range(listLength):
        for key in sampleDict:
            tempList.append( sampleDict[key][index])
        fileWriter.writerow(tempList)
        tempList = []


"""
# NOTES

data = io.loadmat("tasmax_1987_01_01.mat")
#print data['tasmax_1987_01_01'][0][2][0,1,:]

key of dict is name of the file
    data['tasmax_1987_01_01'][0] <-- first array
                                 <-- inside this is three arrays
    [0] = latitude
    [1] = longitude
    [2] = data grid (x, y, time)

    same latitute for each set of 2 points

    0,0 gets top left of grid
location NYC

    take the top left one (to avoid getting over the water temps)
"""