''' The class shall calculate Mean, Mediam, Mode, Variance and Standard Deviation
of Sample data '''
import sys
import numpy as np
from scipy import stats

class StatisticsCalculator:
    def __init__(self):
        super().__init__()

    def mean(sampleDataset):
        return np.mean(sampleDataset)
    
    def median(sampleDataset):
        return np.median(sampleDataset)
    
    def mode(sampleDataset):
        return stats.mode(sampleDataset)[0]
    
    def maximumValue(sampleDataset):
        return np.max(sampleDataset)
    
    def minimumValue(sampleDataset):
        return np.min(sampleDataset)

sys.path.append(".")