# --------------
# --- IMPORT ---
# --------------

import random
import pandas as pd
from machawai.ml.data import InformedTimeSeries, NumericFeature, SeriesFeature

# ---------------
# --- CLASSES ---
# ---------------

class Transformer():

    def __init__(self) -> None:
        pass

    def transform(self, its: InformedTimeSeries) -> InformedTimeSeries:
        raise NotImplementedError()
    
class MinMaxFetaureNormalizer(Transformer):

    def __init__(self, features:'list[str]', df: pd.DataFrame, inplace: bool = False) -> None:
        super().__init__()
        self.features = features
        self.df = df
        self.inplace = inplace

    def min(self, fname: str):
        return self.df.loc["min", fname]
    
    def max(self, fname: str):
        return self.df.loc["max", fname]

    def transform(self, its: InformedTimeSeries) -> InformedTimeSeries:
        if not self.inplace:
            its = its.copy()
        for fname in self.features:
            feat = its.getFeature(fname)
            if feat.isCategorical():
                fval = feat.encode()
                fval = (fval - self.min(fname)) / (self.max(fname) - self.min(fname))
                nfeat = NumericFeature(name=fname, value=fval)
                its.dropFeature(fname)
                its.addFeature(nfeat)
            else:
                feat.value = (feat.value - self.min(fname)) / (self.max(fname) - self.min(fname))
        return its
    
class MinMaxSeriesNormalizer(Transformer):

    def __init__(self, df: pd.DataFrame, inplace: bool = False, colnames: 'list[str]' = None) -> None:
        super().__init__()
        self.df = df
        self.inplace = inplace
        self.colnames = colnames

    def min(self, colname: str):
        return self.df.loc["min", colname]
    
    def max(self, colname: str):
        return self.df.loc["max", colname]

    def transform(self, its: InformedTimeSeries) -> InformedTimeSeries:
        if not self.inplace:
            its = its.copy()
        if self.colnames == None:
            cols = its.getColnames()
        else:
            cols = self.colnames
        for col in cols:
            its.series[col] = (its.series[col] - self.min(col)) / (self.max(col) - self.min(col))
        return its
    
class CutSeriesToMaxIndex(Transformer):

    def __init__(self, 
                 colname: str, 
                 include_features: 'list[str]' = [], 
                 inplace: bool = False) -> None:
        super().__init__()
        self.colname = colname
        self.include_features = include_features
        self.inplace = inplace

    def transform(self, its: InformedTimeSeries) -> InformedTimeSeries:
        if not self.inplace:
            its = its.copy()
        idx_max = its.series[self.colname].argmax()
        its.series = its.series[:idx_max]
        for fname in self.include_features:
            feat = its.getFeature(fname)
            if not isinstance(feat, SeriesFeature):
                raise ValueError("CutSeriesToMax: only SeriesFeature can be transformed.")
            feat.value = feat.value[:idx_max]
        return its

class CutSeriesTail(Transformer):

    def __init__(self, 
                 tail_p: float = 0.0, 
                 include_features: 'list[str]' = [], 
                 use_feature: str = None, 
                 inplace: bool = False) -> None:
        super().__init__()
        self.use_feature = use_feature
        self.tail_p = tail_p
        self.include_features = include_features
        self.inplace = inplace

    def getTailP(self, its: InformedTimeSeries) -> InformedTimeSeries:
        if self.use_feature != None:
            feat = its.getFeature(self.use_feature)
            if isinstance(feat, NumericFeature):
                return feat.value
            raise TypeError("CutSeriesTail: only NumericFeature can be used to cut the series.")
        return self.tail_p

    def transform(self, its: InformedTimeSeries) -> InformedTimeSeries:
        if not self.inplace:
            its = its.copy()
        tail_p = self.getTailP(its=its)

        series_length = its.series.shape[0]
        to_cut = int(series_length * tail_p)      
        its.series = its.series.iloc[:series_length - to_cut]

        for fname in self.include_features:
            feat = its.getFeature(fname)
            if not isinstance(feat, SeriesFeature):
                raise ValueError("CutSeriesTail: only SeriesFeature can be transformed.")
            series_length = feat.value.shape[0]
            to_cut = int(series_length * tail_p)   
            feat.value = feat.value.iloc[:series_length - to_cut]
        return its

class CutSeries(Transformer):

    def __init__(self, cut_size: int = None, min_size: int = 1, max_size: int = 100, inplace: bool = False) -> None:
        super().__init__()
        self.cut_size = cut_size
        self.min_size = min_size
        self.max_size = max_size
        self.inplace = inplace

    def transform(self, its: InformedTimeSeries) -> InformedTimeSeries:
        if not self.inplace:
            its = its.copy()
        series_length = its.series.shape[0]
        if self.cut_size == None:
            cut_size = random.randint(self.min_size, self.max_size)
        else:
            cut_size = self.cut_size
        start_point = random.randint(0, series_length - cut_size)
        its.series = its.series.loc[start_point: start_point + cut_size - 1]
        return its
