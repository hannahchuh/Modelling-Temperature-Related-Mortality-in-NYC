import csv
import numpy

sampleList = [1,1,2,5,2,2,2,2,4,4,4,4,2,2,3,3,4,5,6]

#write list vertically
listLength = len(sampleList)
with open("test.csv", "wb") as fileObj:
    fileWriter = csv.writer(fileObj)
    for index in range(listLength):
        fileWriter.writerow([sampleList[index]])

#write dictionary with lists vertical
sampleDict = {}
for i in range(10):
    sampleList = []
    for j in range(1,11):
        sampleList.append(i*10+j)
    sampleDict.update({str(i+1):sampleList})

#assuming all lists in the dict are of equal length
dictLength = len(sampleDict)
tempList = []
with open("test.csv", "wb") as fileObj:
    fileWriter = csv.writer(fileObj)
    listLength = len(sampleDict.itervalues().next())
    for index in range(listLength):
        for key in sampleDict:
            tempList.append( sampleDict[key][index])
        fileWriter.writerow(tempList)
        tempList = []


#vers if not all lists in the dict are of same len
with open("testAbsMins.csv", "wb") as fileObj:
    fileWriter = csv.writer(fileObj)
    listLength = len(sampleDict.itervalues().next())
    for index in range(listLength):
        for key in sampleDict:
            if index < len(sampleDict[key]):
                tempList.append( sampleDict[key][index])
        fileWriter.writerow(tempList)
        tempList = []