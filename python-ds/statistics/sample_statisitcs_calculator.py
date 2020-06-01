''' The class shall calculate Mean, Mediam, Mode, Variance and Standard Deviation
of Sample data '''
import sys
import numpy as np

class StatisticsCalculator:
    def __init__(self):
        super().__init__()

    def mean(sampleDataset):
        return np.mean(sampleDataset)
    
    def median(sampleDataset):
        return np.median(sampleDataset)
    
    def mode(sampleDataset):
        return np.mode(sampleDataset)

sys.path.append(".")