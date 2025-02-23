# Deep Sort ReID

![Banner Image](https://github.com/cajmorgan/deep_sort_reid/blob/master/material/banner.png)


*API Docs is not yet available, but it's on the way!* 

**Navigation**
- [Deep Sort ReID](#deep-sort-reid)
  - [Introduction](#introduction)
  - [Quick Start](#quick-start)

## Introduction
Deep Sort ReID is an adaption of the prominent 'Deep Sort' Algorithm introduced in 2017, through the following [repo](https://github.com/nwojke/deep_sort) and [paper](https://arxiv.org/pdf/1703.07402). After all these years, Deep Sort still stand out today as a relative simple, modular, lightweight and efficient tracking algorithm. Unfortunately, the original implementation has been abandoned for many years, and usually requires some customisation to make it work with a modern versions of Python and its dependencies. 

![Tracking example of 2 persons](https://github.com/cajmorgan/deep_sort_reid/blob/master/material/walking.webp)


This particular implementation of Deep Sort is an attempt to create a better codebase and add some interesting features to the original algorithm. A common issue with many open-source projects in Python is that they are completely untyped, which hurts debugging and readability. Thus, this project is shipped with Pydantic types, for data validation and usability. The original implementation also had a somewhat messy structure, where many methods didn't feel like they belonged, and other parts being somewhat hard to read. As a result, efforts have been made to improve it. We also use PyTorch instead of NumPy for most parts. 

Something obviously missing from the initial attempt of this algorithm is re-identification, which is also clearly stated. With the inherited modularity of this algorithm, it's not particularly hard to add some basic support for re-identifiation. This is useful when a tracker gets obscured by some other object and then re-appears, as the algorithm can use the features of it to re-assign its previous tracker id without creating a new one. Though, I believe those parts of the project comes with room for improvements. This can be seen in the following example. Note how the multiple of the selected objects gets obscured and re-appear in the frame, and succesfully re-identified.

![ReID in action](https://github.com/cajmorgan/deep_sort_reid/blob/master/material/walking-reid.webp)

Unfortunately, this doesn't work perfectly for all situations, and it heavily depends of the selected hyperparameters, and the input of the algorithm, namely the quality of the detection boxes and extracted features. 

Anyway, a long term goal is to keep improving on this tracker and add more hyperparameters and features, so that it can continue to excell as a lightweight and modular tracker option for many Computer Vision projects. 

![Cars](https://github.com/cajmorgan/deep_sort_reid/blob/master/material/cars.webp)

## Quick Start

To start, use:
```bash
pip install deep_sort_reid
```

In the [example.py](./example.py) file, you can find a general flow that uses YOLO for object detection, and then a metric learning model as a feature extractor on the detected objects. Those are then passed into the DeepSortTracker class, and for the given results, we plot them on the video with cv2. 

Given detections and features, we can simply call the library as:

```python
from deep_sort_reid.DeepSortTracker import DeepSortTracker
from deep_sort_reid.types.tracker import TrackResult
from deep_sort_reid.types.detection import Detection
from typing import List

# First dimension is frame index and second is object within that frame
# Here we assume that features as Torch tensors are added to every detection
detections: List[List[Detection]] = [...]

deep_sort_reid = DeepSortTracker(metric_type="cosine")
track_results: List[List[TrackResult]] = deep_sort_reid.track(detections)
```



