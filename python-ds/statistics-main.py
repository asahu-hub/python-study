import csv
from statistics.sample_statisitcs_calculator import StatisticsCalculator

with open('dataset/Dataset.csv', 'r') as csvfile:
    dataset=csv.DictReader(csvfile)
    dictDataset=list(dataset)
    print('Total number of records: %d' % len(dictDataset))
    print('Headers: ', len(dictDataset[0].keys()))
    