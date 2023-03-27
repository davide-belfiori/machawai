# --------------
# --- IMPORT ---
# --------------

import pandas as pd
from machawai.stats import DataStats
from machawai.ml.data import DataPoint
from machawai.const import LOAD, STRESS

# ---------------
# --- CLASSES ---
# ---------------

class Transformer():
    """
    Base Class for data transformer implementation. 
    """
    def __init__(self) -> None:
        pass

    def __call__(self, data_point: DataPoint) -> DataPoint:
        raise NotImplementedError()

class MinMaxNormalizer(Transformer):
    """
    Apply Min-Max Normalization to a given DataPoint.
    """
    def __init__(self, stats: DataStats, inplace: bool = False, target_feature_map: dict = None) -> None:
        super().__init__()
        self.stats = stats
        self.inplace = inplace
        self.target_feature_map = target_feature_map

        self.min = self.stats.getMin()
        self.max = self.stats.getMax()

    def __call__(self, data_point: DataPoint) -> DataPoint:
        if not self.inplace:
            data_point = data_point.copy()
        # 1) Normalize curve
        if isinstance(data_point.curve, pd.DataFrame):
            for feature in data_point.curve.columns:
                data_point.curve[feature] = (data_point.curve[feature] - self.min[feature]) / (self.max[feature] - self.min[feature])
        else:
            feature = data_point.curve.name
            data_point.curve = (data_point.curve - self.min[feature]) / (self.max[feature] - self.min[feature])
        # 2) Normalize target
        if isinstance(data_point.target, pd.DataFrame):
            for feature in data_point.target.columns:
                if self.target_feature_map != None:
                    mapped_feature = self.target_feature_map[feature]
                    data_point.target[feature] = (data_point.target[feature] - self.min[mapped_feature]) / (self.max[mapped_feature] - self.min[mapped_feature])
                else:
                    data_point.target[feature] = (data_point.target[feature] - self.min[feature]) / (self.max[feature] - self.min[feature])
        else:
            feature = data_point.target.name
            if self.target_feature_map != None:
                mapped_feature = self.target_feature_map[feature]
                data_point.target = (data_point.target - self.min[mapped_feature]) / (self.max[mapped_feature] - self.min[mapped_feature])
            else:
                data_point.target = (data_point.target - self.min[feature]) / (self.max[feature] - self.min[feature])
        return data_point

class MaxLoadCut(Transformer):
    """
    Takes only curve and target values before the maximum load/stress. 
    """
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def __call__(self, data_point: DataPoint) -> DataPoint:
        if not self.inplace:
            data_point = data_point.copy()
        # 1) Cut curve
        if isinstance(data_point.curve, pd.DataFrame):
            if LOAD in data_point.curve.columns:
                idx_max = data_point.curve[LOAD].argmax()
            elif STRESS in data_point.curve.columns:
                idx_max = data_point.curve[STRESS].argmax()
            else:
                raise ValueError("Missing {}/{} feature in input data.".format(LOAD, STRESS))
            data_point.curve = data_point.curve[:idx_max]
        else:
            if data_point.curve.name in [LOAD, STRESS]:
                idx_max = data_point.curve.argmax()
                data_point.curve = data_point.curve[:idx_max]
            else: 
                raise ValueError("Missing {}/{} feature in input data.".format(LOAD, STRESS))
        # 2) Cut target
        data_point.target = data_point.target[:idx_max]
        return data_point
