

from typing import List, Tuple

from deep_sort_reid.enums.tracker import TrackState
from deep_sort_reid.metric.GatedMetric import GatedMetric
from deep_sort_reid.metric.IouMetric import IouMetric
from deep_sort_reid.storage.CacheStorage import CacheStorage
from deep_sort_reid.models.motion.KalmanFilter import KalmanFilter
from deep_sort_reid.trackers.tracktor.Track import Track
from deep_sort_reid.types.detection import Detection
from deep_sort_reid.utils.box_methods import from_xyxy_to_xyah
from deep_sort_reid.utils.linear_assignment import min_cost_matching


class Tracker():

    def __init__(self,
                 gated_metric: GatedMetric,
                 iou_metric: IouMetric,
                 cache_storage: CacheStorage,
                 kf: KalmanFilter,
                 max_since_update: int,
                 hits_until_confirm: int,
                 new_track_max_iou,
                 reid: bool,
                 reid_similarity_score: float,
                 ):

        self.gated_metric = gated_metric
        self.iou_metric = iou_metric
        self.cache_storage = cache_storage
        self.max_since_update = max_since_update
        self.hits_until_confirm = hits_until_confirm
        self.new_track_max_iou = new_track_max_iou
        self.reid = reid
        self.reid_similarity_score = reid_similarity_score

        self.kf = kf
        self.tracks: List[Track] = []
        self.deleted_tracks: List[Track] = []
        self.next_tracker_id = 1

    def __initiate_track(self, detection: Detection, track_id: int = -1):
        new_track_id = self.next_tracker_id
        if track_id != -1:
            new_track_id = track_id

        state_mean, state_covariance = self.kf.initiate(
            from_xyxy_to_xyah(detection.coords))
        new_track = Track(
            state_mean,
            state_covariance,
            new_track_id,
            self.hits_until_confirm,
            self.max_since_update,
            detection.feature,
            detection.cls,
        )

        self.tracks.append(new_track)
        self.next_tracker_id += 1

    def predict(self):
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections_i: List[Detection]):
        confirmed_tracks_idx = []
        unconfirmed_tracks_idx = []
        deleted_tracks_idx = []

        for track_idx, track in enumerate(self.tracks):
            if track.state == TrackState.CONFIRMED:
                confirmed_tracks_idx.append(track_idx)
            elif track.state == TrackState.UNCONFIRMED:
                unconfirmed_tracks_idx.append(track_idx)
            else:
                deleted_tracks_idx.append(track_idx)

        # matches_track_det_idx, \
        #     unmatched_tracks_idx, \
        #     unmatched_detections_idx = \
        #     self.__match(detections_i, confirmed_tracks_idx,
        #                  unconfirmed_tracks_idx)

        # # Update the state of the matched tracks with new measurement information
        # for track_idx, detection_idx in matches_track_det_idx:
        #     self.tracks[track_idx].update(
        #         self.kf,
        #         detections_i[detection_idx])

        # for detection_idx in unmatched_detections_idx:
        #     detection = detections_i[detection_idx]
        #     if self.reid:
        #         self.__match_reid(detection, detection_idx,
        #                           matches_track_det_idx)

        #     # We use the __iou_detection_tracks_check to avoid initating a new track
        #     # in a position where another one already resides
        #     elif self.__iou_detection_tracks_check(detection):
        #         self.__initiate_track(detection)

        # # Potentially remove tracks
        # for track_idx in unmatched_tracks_idx:
        #     self.tracks[track_idx].no_match()

        # for track_idx, detection_idx in matches_track_det_idx:
        #     track = self.tracks[track_idx]
        #     detection = detections_i[detection_idx]

        #     if detection.feature is not None:
        #         self.cache_storage.add_sample(
        #             track.track_id, detection.feature)

        # tracks_to_keep: List[Track] = []
        # for track in self.tracks:
        #     if (track.state != TrackState.DELETED):
        #         tracks_to_keep.append(track)
        #     else:
        #         self.deleted_tracks.append(track)

        # self.tracks = tracks_to_keep

    # def __match_reid(self, detection: Detection, detection_idx: int, matches_track_det_idx: List[Tuple]):
    #     # Potentially re-assign deleted tracks based on visual features
    #     deleted_score, deleted_track_idx = self.__similarity_detection_tracks_match(
    #         detection, self.deleted_tracks)

    #     if deleted_score > self.reid_similarity_score:
    #         re_assigned_track_id = self.deleted_tracks[deleted_track_idx].track_id
    #         self.__initiate_track(
    #             detection, re_assigned_track_id)
    #         re_assigned_track_idx = len(self.tracks) - 1
    #         matches_track_det_idx.append(
    #             (re_assigned_track_idx, detection_idx))

    #         del self.deleted_tracks[deleted_track_idx]

    #     elif self.__iou_detection_tracks_check(detection):
    #         self.__initiate_track(detection)

    # def __similarity_detection_tracks_match(self, detection: Detection, tracks: List[Track]):
    #     highest_similarity_score = 0
    #     highest_similarity_track_idx = -1

    #     for track_idx, track in enumerate(tracks):
    #         track_samples = None

    #         if track.track_id in self.cache_storage.samples:
    #             track_samples = self.cache_storage.get(track.track_id, 'mean')

    #         if (track_samples is not None and detection.feature is not None):
    #             # 1 - in order to transform it back to similarity from distance
    #             # Should be setup to work with euclidean distance likewise
    #             similarity_score = (1 - GatedMetric.cosine_distance(
    #                 track_samples, [detection.feature])).item()

    #             if similarity_score > highest_similarity_score:
    #                 highest_similarity_score = similarity_score
    #                 highest_similarity_track_idx = track_idx

    #     return highest_similarity_score, highest_similarity_track_idx

    # def __iou_detection_tracks_check(self, detection: Detection):
    #     highest_iou = 0

    #     tracks_to_check = self.tracks.copy()

    #     if self.reid:
    #         for track in self.deleted_tracks:
    #             if track.time_since_update < 3:
    #                 tracks_to_check.append(track)

    #     for track in tracks_to_check:
    #         iou = IouMetric.iou(detection.coords, track.get_position())

    #         if iou > highest_iou:
    #             highest_iou = iou
    #     if highest_iou >= self.new_track_max_iou:
    #         return False

    #     return True

    # def __match(self, detections_i: List[Detection], confirmed_tracks_idx: List[int], unconfirmed_tracks_idx: List[int]):
    #     matches_features_track_det_idx, \
    #         unmatched_features_tracks_idx, \
    #         unmatched_features_detections_idx = \
    #         self.__match_features_max_since_update(detections_i,
    #                                                confirmed_tracks_idx)

    #     unmatched_tracks_idx_age_one = []
    #     unmatched_tracks_idx_older = []

    #     for track_idx in unmatched_features_tracks_idx:
    #         if self.tracks[track_idx].time_since_update > 1:
    #             unmatched_tracks_idx_older.append(track_idx)
    #         else:
    #             unmatched_tracks_idx_age_one.append(track_idx)

    #     iou_track_candidates = unconfirmed_tracks_idx + unmatched_tracks_idx_age_one
    #     unmatched_features_tracks_idx = unmatched_tracks_idx_older

    #     matches_iou_track_det_idx, \
    #         unmatched_iou_tracks_idx, \
    #         unmatched_iou_detections_idx = \
    #         self.__match_iou(detections_i, iou_track_candidates,
    #                          unmatched_features_detections_idx)

    #     matches_track_det_idx = matches_features_track_det_idx + matches_iou_track_det_idx
    #     unmatched_tracks_idx: List[int] = list(
    #         set(unmatched_features_tracks_idx + unmatched_iou_tracks_idx))
    #     unmatched_detections_idx = unmatched_iou_detections_idx

    #     return matches_track_det_idx, unmatched_tracks_idx, unmatched_detections_idx

    # def __match_features_max_since_update(self,
    #                                       detections: List[Detection],
    #                                       tracks_idx: List[int]):
    #     """
    #     Matches detections from the current frame (i) to existing tracks that have been updated within
    #     the `max_since_update` range. This method is primarily for re-matching objects that are in-frame.
    #     """

    #     # The detections_idx we haven't matched yet
    #     detections_idx = []

    #     for detection_idx in range(len(detections)):
    #         detections_idx.append(detection_idx)

    #     matches_track_det_idx: List[Tuple[int, int]] = []
    #     unmatched_detections_idx: List[int] = detections_idx

    #     # Note that we start from tracks of age 1 (frame)
    #     for age in range(1, self.max_since_update):
    #         if len(detections_idx) == 0:
    #             break

    #         tracks_idx_curr_age = []
    #         for track_idx in tracks_idx:
    #             if self.tracks[track_idx].time_since_update == age:
    #                 tracks_idx_curr_age.append(track_idx)

    #         if len(tracks_idx_curr_age) == 0:
    #             continue

    #         matches_track_det_idx_curr_age, \
    #             _, \
    #             unmatched_detections_idx = min_cost_matching(self.gated_metric,
    #                                                          self.tracks,
    #                                                          detections,
    #                                                          tracks_idx_curr_age,
    #                                                          unmatched_detections_idx)

    #         matches_track_det_idx.extend(matches_track_det_idx_curr_age)

    #     # We need to make sure to return all tracks_idx that weren't matched
    #     unmatched_tracks_idx = list(set(
    #         tracks_idx) - set(matched_track_idx for matched_track_idx, _ in matches_track_det_idx))

    #     return matches_track_det_idx, unmatched_tracks_idx, unmatched_detections_idx

    # def __match_iou(self, detections: List[Detection], tracks_idx: List[int], detections_idx: List[int]):
    #     matches_track_det_idx, \
    #         unmatched_tracks_idx, \
    #         unmatched_detections_idx = \
    #         min_cost_matching(self.iou_metric, self.tracks,
    #                           detections, tracks_idx, detections_idx)

    #     return matches_track_det_idx, unmatched_tracks_idx, unmatched_detections_idx
