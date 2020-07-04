''' The class shall calculate Mean, Mediam, Mode, Variance and Standard Deviation
of Sample data '''
import sys
import numpy as np
from scipy import stats

class UnivariateStatisticsCalculator:
    def __init__(self):
        super().__init__()

    def mean(sample):
        return np.mean(sample)
    
    def median(sample):
        return np.median(sample)
    
    def mode(sample):
        return stats.mode(sample)[0]
    
    def maximumValue(sample):
        return np.max(sample)
    
    def minimumValue(sample):
        return np.min(sample)

    def variance(sample):
        return np.var(sample)
    
    def standardDeviation(sample):
        return np.std(sample)

sys.path.append(".")