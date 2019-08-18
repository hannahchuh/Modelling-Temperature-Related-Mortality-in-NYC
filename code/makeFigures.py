import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import datetime
from decimal import Decimal
import math

#Functions
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

# import data
shortDict = pickle.load(open("shortCompiledNY.csv", 'rb'))
shortMort = shortDict['mortality']
shortYear = shortDict['year']
shortMonth = shortDict['month']
shortDay = shortDict['day']

M_LAG=30

# convert to datetime objects
datetimeList = []
for i in range(len(shortMort)):
    datetimeList.append(datetime.datetime(shortYear[i], shortMonth[i], shortDay[i]))

#create smoothed mortality
smoothedMortality = rollingAvg(M_LAG, shortMort)

# plot
blue_patch = mpatches.Patch(color='blue', label='Observed mortality')
red_patch = mpatches.Patch(color='red', label="Smoothed mortality")
plt.legend(handles=[blue_patch, red_patch])

plt.title("Daily observed mortality", fontsize = 15)
plt.xlabel('Year', fontsize=15)
plt.ylabel('Daily mortality', fontsize=15)
plt.plot(datetimeList, shortMort)
plt.plot(datetimeList, smoothedMortality,color='red')
plt.show()