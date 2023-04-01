"""
    Classes and functions for pytorch integration.
"""
# --------------
# --- IMPORT ---
# --------------

from machawai.ml.data import InformedTimeSeries, Feature
from machawai.ml.transformer import Transformer
import torch
import numpy as np

# ---------------
# --- CLASSES ---
# ---------------

class ITSWrapper(InformedTimeSeries):

    def __init__(self, its: InformedTimeSeries, add_batch_dim: bool = True, device: str = "cpu", dtype = torch.float32) -> None:
        self.its = its
        self.device = device
        self.dtype = dtype
        self.add_batch_dim = add_batch_dim

    def getColnames(self):
        return self.its.getColnames()
    
    def hasColumn(self, name: str):
        return self.its.hasColumn(name)

    def getColumn(self, name:str):
        return self.its.getColumn(name)

    def getFeature(self, name:str):
        return self.its.getFeature(name)
    
    def hasFeature(self, name:str):
        return self.its.hasFeature(name)
    
    def dropFeature(self, name:str):
        self.its.dropFeature(name)

    def addFeature(self, feature: Feature):
        self.its.addFeature(feature)

    def hasTarget(self, name:str):
        return self.its.hasTarget(name)

    def getTarget(self, name:str):
        return self.its.getTarget(name)
    
    def untarget(self, name:str):
        self.its.untarget(name)

    def setTarget(self, name: str):
        self.its.setTarget(name)
    
    def copySeries(self):
        return self.its.copySeries()

    def copyFeatures(self):
        return self.its.copyFeatures()

    def getTrainSeries(self):
        return self.its.getTrainSeries()
    
    def getTargetSeries(self):
        return self.its.getTargetSeries()
    
    def getTrainFeatures(self):
        return self.its.getTrainFeatures()
    
    def getTargetFeatures(self):
        return self.its.getTargetFeatures()

    def copy(self) -> 'ITSWrapper':
        return ITSWrapper(its=self.its.copy(),
                          add_batch_dim=self.add_batch_dim,
                          device=self.device,
                          dtype=self.dtype)

    def train_series_tensor(self, add_batch_dim: bool = False):
        train_series = self.getTrainSeries().values
        tensor = torch.tensor(train_series, device=self.device, dtype=self.dtype)
        if add_batch_dim:
            tensor = tensor[None, ...]
        return tensor
    
    def train_features_tensor(self, add_batch_dim: bool = False):
        features = np.array([])
        for i, feat in enumerate(self.its.getTrainFeatures()):
            if i == 0:
                features = feat.encode()
            else:
                features = np.hstack([features, feat.encode()])
        if features.any():
            features = torch.tensor(features, device=self.device, dtype=self.dtype)
            if add_batch_dim:
                features = features[None, ...]
        else:
            features = torch.tensor([], device=self.device)
        return features
    
    def target_tensor(self, add_batch_dim: bool = False):
        target = []
        # Append series tensor
        s_tensor = self.getTargetSeries().values
        s_tensor = s_tensor.squeeze()
        s_tensor = torch.tensor(s_tensor, device=self.device, dtype=self.dtype)
        if add_batch_dim:
            s_tensor = s_tensor[None, ...]
        target.append(s_tensor)
        # Append feature tensors
        for ft in self.getTargetFeatures():
            f_tensor = ft.encode()
            f_tensor = torch.tensor(f_tensor, device=self.device, dtype=self.dtype)
            if add_batch_dim:
                f_tensor = f_tensor[None, ...]
            target.append(f_tensor)
        if len(target) == 1:
            return target[0]
        return target

    def X(self):
        train_series = self.train_series_tensor(add_batch_dim=self.add_batch_dim)
        train_features = self.train_features_tensor(add_batch_dim=self.add_batch_dim)
        if train_features.any():
            return train_series, train_features
        return train_series
    
    def Y(self):
        return self.target_tensor(self.add_batch_dim)

    def __iter__(self):
        return iter((self.X(), self.Y()))

class WrapTransformer(Transformer):

    def __init__(self, add_batch_dim: bool = True, device: str = "cpu", dtype = torch.float32) -> None:
        super().__init__()
        self.add_batch_dim = add_batch_dim
        self.device = device
        self.dtype = dtype

    def transform(self, its: InformedTimeSeries) -> InformedTimeSeries:
        return ITSWrapper(its=its, 
                          add_batch_dim=self.add_batch_dim,
                          device=self.device,
                          dtype=self.dtype)
