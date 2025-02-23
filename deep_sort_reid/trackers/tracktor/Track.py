

from typing import List, Sequence
from torch import Tensor
from deep_sort_reid.enums.tracker import TrackState
from deep_sort_reid.models.motion.KalmanFilter import KalmanFilter
from deep_sort_reid.types.coords import CoordinatesXYXY
from deep_sort_reid.types.detection import Detection

from deep_sort_reid.types.tracker import TrackID
from deep_sort_reid.utils.box_methods import from_xyxy_to_xyah


class Track():

    def __init__(self, state_mean,
                 state_covariance,
                 track_id,
                 hits_until_confirm,
                 max_since_update,
                 feature=None,
                 cls=[],
                 ):

        self.state_mean: Tensor = state_mean
        self.state_covariance: Tensor = state_covariance
        self.track_id: TrackID = track_id
        self.hits: int = 1
        self.time_since_update: int = 0
        self.hits_until_confirm: int = hits_until_confirm
        self.max_since_update: int = max_since_update
        self.cls: List[int] = cls

        self.state = TrackState.UNCONFIRMED

    def predict(self, kf: KalmanFilter):
        self.state_mean, self.state_covariance = kf.predict(
            self.state_mean, self.state_covariance)

        self.time_since_update += 1

    def update(self, kf: KalmanFilter, detection: Detection):
        self.state_mean, self.state_covariance = kf.update(self.state_mean,
                                                           self.state_covariance,
                                                           from_xyxy_to_xyah(detection.coords))

        self.hits += 1
        self.time_since_update = 0

        if self.state == TrackState.UNCONFIRMED and self.hits >= self.hits_until_confirm:
            self.state = TrackState.CONFIRMED

    def no_match(self):

        if self.state == TrackState.UNCONFIRMED:
            self.state = TrackState.DELETED
            return True
        elif self.time_since_update > self.max_since_update:
            self.state = TrackState.DELETED
            return True

        return False

    def get_position(self):
        center_x, center_y, aspect_ratio, height = self.state_mean[:4]

        width = aspect_ratio * height
        start_x = (center_x - (width // 2)).item()
        end_x = (start_x + width).item()
        start_y = (center_y - (height // 2)).item()
        end_y = (start_y + height).item()

        state_coords = CoordinatesXYXY(start_x=start_x, end_x=end_x,
                                       start_y=start_y, end_y=end_y)

        return state_coords
