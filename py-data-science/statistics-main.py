import csv
from statistics.sample_statisitcs_calculator import StatisticsCalculator
import matplotlib.pyplot as plt

with open('dataset/Dataset.csv', 'r') as csvfile:
    dataset=csv.DictReader(csvfile)
    dictDataset=list(dataset)
    header_value='VALUE'
    values=[]
    for datapoint in dictDataset:
        cValue=datapoint[header_value]
        if(cValue != ''):
            values.append(float(cValue))
    
    print('Minimum Value: ', StatisticsCalculator.minimumValue(values), '\nMaximum Value: ', StatisticsCalculator.maximumValue(values),'\nMean: ', StatisticsCalculator.mean(values), '\nMedian: ', StatisticsCalculator.median(values),'\nMode: ', StatisticsCalculator.mode(values))

    valueIndexes=range(0, len(values))
    plt.scatter(valueIndexes, values)
    plt.show()