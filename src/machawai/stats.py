"""
Tensile Test Statistics.
"""

import pandas as pd
from machawai.tensile import TensileTest

class DataStats():
    """
    Compute data statistics on a collection of `TensileTest` objects.
    """
    def __init__(self, tests: 'list[TensileTest]') -> None:
        assert len(tests) > 0
        self.tests = tests

        self.data_labels = self.tests[0].getDataLabels()
        self.data_stats = list(map(lambda test: test.getDataStats(), self.tests))

    def getMin(self):
        """
        Return the minimum values.
        """
        min_vals = map(lambda ds: ds.loc['min'].values, self.data_stats)
        min_df = pd.DataFrame(min_vals, columns=self.data_labels)
        return min_df.min()

    def getMax(self):
        """
        Return the maximum values.
        """
        max_vals = map(lambda ds: ds.loc['max'].values, self.data_stats)
        max_df = pd.DataFrame(max_vals, columns=self.data_labels)
        return max_df.max()
    
    def getMean(self):
        """
        Return the mean values.
        """
        mean_vals = map(lambda ds: ds.loc['mean'].values, self.data_stats)
        mean_df = pd.DataFrame(mean_vals, columns=self.data_labels)
        return mean_df.mean()

