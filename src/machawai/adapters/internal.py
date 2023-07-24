"""
    Classes and functions for internal packages communication.
"""

# --------------
# --- IMPORT ---
# --------------

from machawai.tensile import TensileTest
from machawai.ml.data import InformedTimeSeries, parse_feature, ITSDatasetConfig, InformedTimeSeriesDataset, VectorFeature
from machawai.ml.transformer import Transformer
from machawai.labels import *
from scipy import interpolate
import numpy as np
import pandas as pd

# ---------------
# --- CLASSES ---
# ---------------

class CustomInterpolation(Transformer):

    def __init__(self,
                 size: int,
                 disp_label: str = DISPLACEMENT,
                 load_label: str = LOAD,
                 strain_label: str = EXTS_STRAIN,
                 kind: str = 'cubic',
                 save_old_values: 'list[str]' = [DISPLACEMENT, LOAD, EXTS_STRAIN],
                 inplace: bool = False) -> None:
        super().__init__()
        self.size = size
        self.disp_label = disp_label
        self.load_label = load_label
        self.strain_label = strain_label
        self.inplace = inplace
        self.kind = kind
        self.save_old_values = save_old_values

    def transform(self, its: InformedTimeSeries) -> InformedTimeSeries:
        if not self.inplace:
            its = its.copy()
        t = np.arange(0, its.series.shape[0], 1)
        disp = its.getColumn(self.disp_label).values
        load = its.getColumn(self.load_label).values
        strain = its.getColumn(self.strain_label).values

        y = np.vstack([disp[None, ...], load[None, ...]])
        spline = interpolate.interp1d(t, y, kind=self.kind)
        new_t = np.linspace(t.min(), t.max(), self.size)
        new_y = spline(new_t)
        new_disp = new_y[0,:]
        new_load = new_y[1,:]

        y_2 = np.vstack([strain[None, ...], load[None, ...]])
        spline_2 = interpolate.interp1d(t, y_2, kind=self.kind)
        new_y_2 = spline_2(new_t)
        new_strain = new_y_2[0,:]
        new_load_2 = new_y_2[1,:]

        assert (new_load == new_load_2).sum() == self.size

        new_series = pd.DataFrame({
            self.disp_label : new_disp,
            self.load_label : new_load,
            self.strain_label : new_strain
        })
        for label in self.save_old_values:
            name = "OLD_"+label
            value = its.getColumn(label).tolist()
            its.addFeature(VectorFeature(name = name, value = value))
        its.series = new_series

        return its

# -----------------
# --- FUNCTIONS ---
# -----------------

def parse_tensile_test(ttest: TensileTest, config: ITSDatasetConfig) -> InformedTimeSeries:
    series = ttest.getFullData()
    features = []
    labels = ttest.labels()
    for label in labels:
        if not label in series.columns:
            feature = parse_feature(name=label,
                                    value=ttest.get_by_label(label), 
                                    config=config)
            features.append(feature)

    return InformedTimeSeries(series=series,
                              features=features,
                              data_train=config.data_train,
                              data_target=config.data_target,
                              features_train=config.features_train,
                              features_target=config.features_target,
                              name=ttest.filename)

def parse_tensile_test_collection(collection: 'list[TensileTest]', config: ITSDatasetConfig) -> InformedTimeSeriesDataset:
    data = []
    for ttest in collection:
        its = parse_tensile_test(ttest=ttest, config=config)
        data.append(its)
    dataset = InformedTimeSeriesDataset(data = data, config = config)
    return dataset