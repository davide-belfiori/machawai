"""
Classes and functions for ML data management.
"""

# --------------
# --- IMPORT ---
# --------------

import numpy as np
import pandas as pd
from machawai.tensile import TensileTest
from machawai.ml.utils import flatten_dictionary

# ---------------
# --- CLASSES ---
# ---------------

class DataPoint():

    def __init__(self, 
                 curve: 'pd.Series | pd.DataFrame',
                 info: np.ndarray,
                 target: 'pd.Series | pd.DataFrame') -> None:
        """
        Arguments:
        ----------

        curve: DataFrame
            Curve data points.

        info: ndarray
            Additional info.

        target: pd.Series
            Target
        """
        self.curve = curve
        self.info = info
        self.target = target

    def X(self, squeeze: bool = False) -> tuple:
        if squeeze:
            return (self.curve.values.squeeze(), self.info.squeeze())
        return (self.curve.values, self.info)
    
    def Y(self, squeeze: bool = False) -> np.ndarray:
        if squeeze:
            return self.target.values.squeeze()
        return self.target.values
    
    def point(self, squeeze: bool = False) -> tuple:
        return (self.X(squeeze=squeeze), self.Y(squeeze=squeeze))
    
    def copy(self):
        return DataPoint(curve=self.curve.copy(),
                         info=self.info.copy(),
                         target=self.target.copy())

# -----------------
# --- FUNCTIONS ---
# -----------------

def tensileTest2DataPoint(tensileTest: TensileTest,
                          curve_select: 'list[str]' = None,
                          sprop_select: 'list[str]' = None,
                          target_select: 'list[str]' = None) -> DataPoint:
    """
    Convert a `TensileTest` object into a `DataPoint`.
    """
    if curve_select == None:
        curve = tensileTest.getData()
    else:
        curve = tensileTest.selectData(labels=curve_select)
    if sprop_select == None:
        info = tensileTest.getProperties()
    else:
        info = tensileTest.selectProperties(labels=sprop_select)
    info = flatten_dictionary(info)
    if target_select == None:
        target = tensileTest.getExtsStrain()
    else:
        target = tensileTest.selectData(labels=target_select)
    data_point = DataPoint(curve = curve,
                           info = info,
                           target = target)
    return data_point