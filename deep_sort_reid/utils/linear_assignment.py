

from typing import List, Sequence, Tuple
from deep_sort_reid.metric.IouMetric import IouMetric
from deep_sort_reid.metric.GatedMetric import GatedMetric
from deep_sort_reid.trackers.deep_sort.Track import Track
from deep_sort_reid.types.detection import Detection
from scipy.optimize import linear_sum_assignment


def min_cost_matching(metric: GatedMetric | IouMetric,
                      tracks: List[Track],
                      detections: List[Detection],
                      tracks_idx: List[int],
                      detections_idx: List[int]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:

    if len(detections_idx) == 0 or len(tracks_idx) == 0:
        return [], tracks_idx, detections_idx

    selected_tracks: List[Track] = []
    selected_detections: List[Detection] = []

    for track_idx in tracks_idx:
        selected_tracks.append(tracks[track_idx])

    for detection_idx in detections_idx:
        selected_detections.append(detections[detection_idx])

    cost_matrix = metric(selected_tracks, selected_detections)

    assert cost_matrix is not None

    cost_matrix[cost_matrix > metric.max_distance] = metric.max_distance + 1e-5

    (rows_idx, cols_idx) = linear_sum_assignment(cost_matrix)

    matches_track_det_idx: List[Tuple[int, int]] = []
    unmatched_tracks_idx: List[int] = []
    unmatched_detections_idx: List[int] = []

    for col_idx, detection_idx in enumerate(detections_idx):
        if col_idx not in cols_idx:
            unmatched_detections_idx.append(detection_idx)

    for row_idx, track_idx in enumerate(tracks_idx):
        if row_idx not in rows_idx:
            unmatched_tracks_idx.append(track_idx)

    for idx, row_idx in enumerate(rows_idx):
        track_idx = tracks_idx[row_idx]

        # the column that matches with the row in the assignment
        col_match = cols_idx[idx]
        detection_idx = detections_idx[col_match]

        if cost_matrix[row_idx, col_match] > metric.max_distance:
            unmatched_tracks_idx.append(track_idx)
            unmatched_detections_idx.append(detection_idx)
        else:
            matches_track_det_idx.append((track_idx, detection_idx))

    return matches_track_det_idx, unmatched_tracks_idx, unmatched_detections_idx
