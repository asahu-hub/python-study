from statistics.UnivariateStatistics import UnivariateStatisticsCalculator as UniStats
import csv
import matplotlib.pyplot as plt
import random

with open('dataset/Dataset.csv', 'r') as csvfile:
    dataset=csv.DictReader(csvfile)
    dictDataset=list(dataset)
    header_value='VALUE'
    data=[]
    values=[]
    for datapoint in dictDataset:
        cValue=datapoint[header_value]
        if(cValue != ''):
            data.append(float(cValue))
    
    values=random.choices(data, k=10)

    ## Univariate Statistics
    print('Minimum Value: ', UniStats.minimumValue(values), '\nMaximum Value: ', UniStats.maximumValue(values),'\nMean: ', UniStats.mean(values), '\nMedian: ', UniStats.median(values),'\nMode: ', UniStats.mode(values), '\nVariance: ', UniStats.variance(values), '\nStandard Deviation: ', UniStats.standardDeviation(values))

    valueIndexes=range(0, len(values))
    plt.scatter(valueIndexes, values)
    plt.show()