from typing import List, Sequence

from torch import Tensor
import torch
from deep_sort_reid.trackers.deep_sort.Track import Track
from deep_sort_reid.types.detection import Detection
from deep_sort_reid.types.metric import MetricType
from abc import ABC, abstractmethod


class Metric():

    def __init__(self, metric_type: MetricType, max_distance: float):
        self.metric_type: MetricType = metric_type
        self.max_distance = max_distance

    @abstractmethod
    def distance(self, tracks: List[Track],
                 detections: List[Detection]) -> torch.Tensor:
        """
        Create a cost matrix
        where rows indicate detection targets
        amd columns the features.
        For every row (detection target), calculate
        the distance using the chosen metric, between the 
        current row detection features and the rest of the provided 
        features   
        """

        return torch.Tensor()

    @abstractmethod
    def __call__(self,
                 tracks: List[Track],
                 detections: List[Detection],
                 ) -> Tensor:

        return torch.Tensor()
