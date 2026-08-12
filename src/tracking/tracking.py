import numpy as np
import cv2
from src.config.setting import MIN_CONF, NMS_THRESH
from .deep_sort.detection import Detection


def detect_human(detector_obj, frame, encoder, tracker,frame_time):
	"""
	Detect humans using ultralytics YOLO
	"""
	boxes, confidences, centroids = detector_obj.detect(frame)

	# Apply NMS using OpenCV
	if len(boxes) > 0:
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

		if len(idxs) > 0:
			filtered_boxes = []
			filtered_confidences = []
			filtered_centroids = []

			idxs = np.array(idxs).reshape(-1)
			for i in idxs:
				filtered_boxes.append(boxes[i])
				filtered_confidences.append(confidences[i])
				filtered_centroids.append(centroids[i])

			boxes = np.array(filtered_boxes)
			confidences = np.array(filtered_confidences)
			centroids = np.array(filtered_centroids)

			# Generate features and detections
			features = np.array(encoder(frame, boxes))
			detections = [Detection(bbox, score, centroid, feature)
					 for bbox, score, centroid, feature in zip(boxes, confidences, centroids, features)]

			tracker.predict()
			expired = tracker.update(detections, frame_time)

			tracked_bboxes = []
			for track in tracker.tracks:
				if not track.is_confirmed() or track.time_since_update > 5:
					continue
				tracked_bboxes.append(track)

			return tracked_bboxes, expired

	return [], []

