

from typing import List

from torch import Tensor
import torch
from deep_sort_reid.trackers.deep_sort.Track import Track
from deep_sort_reid.types.coords import CoordinatesXYXY
from deep_sort_reid.types.detection import Detection
from deep_sort_reid.types.metric import MetricType
from deep_sort_reid.metric.Metric import Metric


class IouMetric(Metric):

    def __init__(self, metric_type: MetricType, max_distance: float):
        self.metric_type: MetricType = metric_type
        self.max_distance = max_distance

    def __distance(self,
                   tracks: List[Track],
                   detections: List[Detection],
                   ) -> Tensor:

        cost_matrix = torch.zeros((len(tracks)), len(detections))

        for idx, track in enumerate(tracks):
            # if track.time_since_update > 5:
            #     cost_matrix[idx, :] = torch.inf
            #     continue

            track_state_coords: CoordinatesXYXY = track.get_position()

            cost_matrix[idx, :] = self.__iou_distances(
                track_state_coords, detections)

        return cost_matrix

    def __iou_distances(self, track_state_coords: CoordinatesXYXY, detections: List[Detection]):

        distances = []

        for detection in detections:
            iou_distance = 1 - self.iou(track_state_coords, detection.coords)
            distances.append(iou_distance)

        return torch.Tensor(distances)

    @staticmethod
    def iou(box_1: CoordinatesXYXY, box_2: CoordinatesXYXY):
        (box_1_start_x, box_1_end_x,
         box_1_start_y, box_1_end_y) = (min(box_1.start_x, box_1.end_x), max(box_1.start_x, box_1.end_x),
                                        min(box_1.start_y, box_1.end_y), max(box_1.start_y, box_1.end_y))

        (box_2_start_x, box_2_end_x,
         box_2_start_y, box_2_end_y) = (min(box_2.start_x, box_2.end_x), max(box_2.start_x, box_2.end_x),
                                        min(box_2.start_y, box_2.end_y), max(box_2.start_y, box_2.end_y))

        intersection_width = max(0, min(
            box_1_end_x, box_2_end_x) - max(box_1_start_x, box_2_start_x))

        intersection_height = max(0, min(
            box_1_end_y, box_2_end_y) - max(box_1_start_y, box_2_start_y))

        intersection_area = intersection_width * intersection_height

        box_1_area = (box_1_end_x - box_1_start_x) * \
            (box_1_end_y - box_1_start_y)

        box_2_area = (box_2_end_x - box_2_start_x) * \
            (box_2_end_y - box_2_start_y)

        union_area = box_1_area + box_2_area - intersection_area

        iou = (intersection_area / union_area)
        return iou

    def __call__(self,
                 tracks: List[Track],
                 detections: List[Detection],
                 ):

        return self.__distance(tracks, detections)
