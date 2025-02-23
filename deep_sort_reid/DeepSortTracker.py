

"""
What to pass in:

- features as Tensors or ndarrays, indexed by frames, indexed by objects
- Sequence of Sequence of Detection, indexed as the features

"""


import random
from typing import List
from deep_sort_reid.metric.GatedMetric import GatedMetric
from deep_sort_reid.metric.IouMetric import IouMetric
from deep_sort_reid.storage.CacheStorage import CacheStorage
from deep_sort_reid.models.motion.KalmanFilter import KalmanFilter
from deep_sort_reid.trackers.deep_sort.Tracker import Tracker
from deep_sort_reid.types.tracker import TrackResult
from deep_sort_reid.types.detection import Detection
from deep_sort_reid.types.metric import MetricType
from pydantic import TypeAdapter


class DeepSortTracker():

    def __init__(self,
                 # Config
                 metric_type: MetricType = "cosine",
                 features_max_distance: float = 0.5,
                 iou_max_distance: float = 0.5,
                 max_since_update: int = 5,
                 max_samples_per_track: int = 30,
                 hits_until_confirm: int = 3,
                 new_track_max_iou=0.5,
                 reid=False,
                 reid_similarity_score=0.9,
                 verbose=False
                 ):
        self.metric_type: MetricType = metric_type
        self.features_max_distance = features_max_distance
        self.iou_max_distance = iou_max_distance
        self.max_samples_per_track = max_samples_per_track
        self.max_since_update = max_since_update
        self.hits_until_confirm = hits_until_confirm
        self.new_track_max_iou = new_track_max_iou
        self.reid = reid
        self.reid_similarity_score = reid_similarity_score
        self.verbose = verbose

    def track(self,
              detections: List[List[Detection]]):

        detections = TypeAdapter(
            List[List[Detection]]).validate_python(detections)

        kf = KalmanFilter()
        cache_storage = CacheStorage(self.max_samples_per_track)
        gated_metric = GatedMetric(
            self.metric_type, self.features_max_distance, cache_storage, kf)
        iou_metric = IouMetric('iou', self.iou_max_distance)

        tracker = Tracker(gated_metric, iou_metric,
                          cache_storage, kf,
                          max_since_update=self.max_since_update,
                          hits_until_confirm=self.hits_until_confirm,
                          new_track_max_iou=self.new_track_max_iou,
                          reid=self.reid,
                          reid_similarity_score=self.reid_similarity_score,
                          )

        no_of_frames = len(detections)
        tracker_results: List[List[TrackResult]] = []
        for _ in range(no_of_frames):
            tracker_results.append([])

        colors = {}

        base_color_bgr = (205, 179, 58)  # BGR values for #3AB3CD

        for frame_idx in range(no_of_frames):
            if self.verbose:
                print("PROCESSEING FRAME: ", frame_idx)

            detections_i = detections[frame_idx]
            tracker.predict()
            tracker.update(detections_i)

            for track in tracker.tracks:
                if track.track_id not in colors:
                    colors[track.track_id] = self.__generate_shade_of_base_color(
                        base_color_bgr)

                track_result = TrackResult(track_id=track.track_id, frame_idx=frame_idx,
                                           coords=track.get_position(), cls=track.cls, state=track.state,
                                           color=colors[track.track_id])

                tracker_results[frame_idx].append(track_result)

        return tracker_results

    def __generate_shade_of_base_color(self, base_color, variation_range=(-30, 30)):
        # Apply variation to the BGR values to generate a lighter or darker shade
        b = min(255, max(0, base_color[0] +
                random.randint(*variation_range)))
        g = min(255, max(0, base_color[1] +
                random.randint(*variation_range)))
        r = min(255, max(0, base_color[2] +
                random.randint(*variation_range)))
        return (b, g, r)
