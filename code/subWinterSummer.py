import pickle
import numpy as np
import matplotlib.pyplot as plt
import csv

modelName = "gfdl"
file = modelName + "ModelMort"
summerDict = pickle.load(open(file+"Summer.csv", 'rb'))
winterDict = pickle.load(open(file+"Winter.csv", 'rb'))

futWinterMort = winterDict['FutureMortality']
futSummerMort = summerDict['FutureMortality']
pastWinterMort = winterDict['HistMortality']
pastSummerMort = summerDict['HistMortality']

# remove first in summer (2080)
futSummerMort = futSummerMort[1:]
pastSummerMort = pastSummerMort[1:]

futAxis = []
futSubWinterSummer= []
pastSubWinterSummer = []

for i in range(len(futSummerMort)):
    if futSummerMort[i] != 'nan' and futWinterMort[i] != 'nan':
        futSubWinterSummer.append(np.float64(futSummerMort[i] - futWinterMort[i]))
    else:
        futSubWinterSummer.append(np.float64('nan'))

for i in range(len(pastSummerMort)):
    if pastSummerMort[i] != 'nan' and pastWinterMort[i] != 'nan':
        pastSubWinterSummer.append(np.float64(pastSummerMort[i] - pastWinterMort[i]))
    else:
        pastSubWinterSummer.append(np.float64('nan'))

# graphing
plt.hold(True)
futAxis = np.arange(2021,2081)
plt.scatter(futAxis, futSubWinterSummer)
pastAxis = np.arange(1988,2001)
plt.scatter(pastAxis, pastSubWinterSummer)
plt.title("Projected summer - winter mortality", fontsize=15)
plt.ylabel("Total mortality anomaly", fontsize = 15)
plt.xlabel("Year", fontsize=15)
plt.show()

"""
# pickle future mortality proj
exportDict = {}
fileName = modelName + "SubWinterSummer.csv"
exportDict.update({'futSubWinterSummer':futSubWinterSummer})
exportDict.update({'pastSubWinterSummer':pastSubWinterSummer})
with open(fileName, 'wb') as handle:
    pickle.dump(exportDict, handle)

# write future mortality proj to readable csv file
sampleDict = exportDict
dictLength = len(sampleDict)
tempList = []
fileName = modelName + "SubWinterSummerReadable.csv"
with open(fileName, "wb") as fileObj:
    fileWriter = csv.writer(fileObj)
    listLength = len(sampleDict.itervalues().next())
    for index in range(listLength):
        for key in sampleDict:
            tempList.append( sampleDict[key][index])
        fileWriter.writerow(tempList)
        tempList = []
"""