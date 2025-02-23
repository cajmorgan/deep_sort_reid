

from typing import List

from torch import Tensor
import torch
from deep_sort_reid.constants.tracker import GATING_THRESHOLD
from deep_sort_reid.storage.CacheStorage import CacheStorage
from deep_sort_reid.models.motion.KalmanFilter import KalmanFilter
from deep_sort_reid.trackers.deep_sort.Track import Track
from deep_sort_reid.types.coords import CoordinatesXYAH
from deep_sort_reid.types.detection import Detection
from deep_sort_reid.types.metric import MetricType
from deep_sort_reid.metric.Metric import Metric
from deep_sort_reid.utils.box_methods import from_xyah_to_tensor, from_xyxy_to_xyah


class GatedMetric(Metric):

    def __init__(self, metric_type: MetricType, max_distance: float, cache_storage: CacheStorage, kf: KalmanFilter, gated: bool = True):
        self.metric_type: MetricType = metric_type
        self.max_distance = max_distance
        self.cache_storage: CacheStorage = cache_storage
        self.kf: KalmanFilter = kf
        self.gated: bool = gated
        # Distance usage -> f.e do we want min over s

    def __distance(self,
                   tracks: List[Track],
                   detections: List[Detection]) -> Tensor:

        cost_matrix = torch.zeros((len(tracks)), len(detections))

        if self.metric_type == 'iou':
            cost_matrix += torch.inf
            return cost_matrix

        features = []
        measurements: List[Tensor] = []
        for detection in detections:
            measurements.append(from_xyah_to_tensor(
                from_xyxy_to_xyah(detection.coords)))
            if detection.feature is None:
                # If features are missing from detection,
                # we simply return it as inf
                cost_matrix += torch.inf
                return cost_matrix

            features.append(detection.feature)

        for track_idx, track in enumerate(tracks):
            if self.metric_type == 'cosine':
                cost_matrix[track_idx, :] = self.cosine_distance(
                    self.cache_storage[track.track_id], features)
            elif self.metric_type == 'euclidean':
                cost_matrix[track_idx, :] = self.euclidean_distance(
                    self.cache_storage[track.track_id], features)

        cost_matrix = self.__gated(cost_matrix, tracks, measurements)
        return cost_matrix

    def __gated(self, cost_matrix: Tensor, tracks: List[Track], measurements: List[Tensor]):
        gating_threshold = GATING_THRESHOLD
        measurements_tensor = torch.stack(measurements)

        for track_idx, track in enumerate(tracks):
            gating_distance = self.kf.gating_distance(
                track.state_mean, track.state_covariance, measurements_tensor)

            cost_matrix[track_idx, gating_distance >
                        gating_threshold] = torch.inf

        return cost_matrix

    @staticmethod
    def cosine_distance(samples: List[Tensor], features: List[Tensor]):
        samples_mat = torch.stack(samples)
        features_mat = torch.stack(features)

        A = samples_mat / torch.linalg.norm(samples_mat, dim=1, keepdim=True)
        B = features_mat / torch.linalg.norm(features_mat, dim=1, keepdim=True)

        distances = torch.min((1 - (A @ B.T)), dim=0).values

        return distances

    @staticmethod
    def euclidean_distance(samples: List[Tensor], features: List[Tensor]):
        samples_mat = torch.stack(samples)
        features_mat = torch.stack(features)

        A = samples_mat.unsqueeze(1)
        B = features_mat.unsqueeze(0)

        diff = A - B

        distances = torch.sqrt(torch.sum(torch.square(diff), dim=2))
        distances = torch.min(distances, dim=0).values

        return distances

    @staticmethod
    def similarity(A: Tensor, B: Tensor):

        A_norm = A / torch.linalg.norm(A)
        B_norm = B / torch.linalg.norm(B)

        return A_norm@B_norm

    def __call__(self,
                 tracks: List[Track],
                 detections: List[Detection],
                 ):

        return self.__distance(tracks, detections)
