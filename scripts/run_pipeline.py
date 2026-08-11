
import datetime
import time
import cv2
from pathlib import Path



from src.config.setting import(YOLO_CONFIG, VIDEO_CONFIG, DATA_RECORD_RATE, FRAME_SIZE, TRACK_MAX_AGE)
from src.detection.yolo_detector import create_yolo_detector
from src.processing.video_process import video_process
from src.tracking.deep_sort import nn_matching
from src.tracking.deep_sort.tracker import Tracker
from src.tracking.deep_sort import generate_detections as gdet
from src.database.csv_storage import CSVStorage

#create root path
ROOT_DIR = Path(__file__).resolve().parents[1]
#create storage
storage = CSVStorage(ROOT_DIR)

# Read from video
IS_CAM = VIDEO_CONFIG["IS_CAM"]
cap = cv2.VideoCapture(VIDEO_CONFIG["VIDEO_CAP"])


model_version = YOLO_CONFIG.get("MODEL_VERSION", "yolov8m")
device = YOLO_CONFIG.get("DEVICE", "cpu")

print(f"Loading {model_version}...")
detector_obj = create_yolo_detector(
	yolo_version=model_version,
	device=device,
)

# Tracker parameters
max_cosine_distance = 0.7
nn_budget = None

#initialize deep sort object
if IS_CAM: 
	max_age = VIDEO_CONFIG["CAM_APPROX_FPS"] * TRACK_MAX_AGE
else:
	max_age=DATA_RECORD_RATE * TRACK_MAX_AGE


tracker_model_filename = (
    ROOT_DIR /
    "models" /
    "pretrained" /
    "mars-small128.pb"
)
encoder = gdet.create_box_encoder(tracker_model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric, max_age=max_age)

START_TIME = time.time()

# Pass detector to video_process
processing_FPS = video_process(cap, FRAME_SIZE, encoder, tracker, storage, detector_obj=detector_obj)

cv2.destroyAllWindows()

END_TIME = time.time()
PROCESS_TIME = END_TIME - START_TIME
print("Time elapsed: ", PROCESS_TIME)
if IS_CAM:
	print("Processed FPS: ", processing_FPS)
	VID_FPS = processing_FPS
	DATA_RECORD_FRAME = 1
else:
	print("Processed FPS: ", round(cap.get(cv2.CAP_PROP_FRAME_COUNT) / PROCESS_TIME, 2))
	VID_FPS = cap.get(cv2.CAP_PROP_FPS)
	DATA_RECORD_FRAME = int(VID_FPS / DATA_RECORD_RATE)
	START_TIME = VIDEO_CONFIG["START_TIME"]
	time_elapsed = round(cap.get(cv2.CAP_PROP_FRAME_COUNT) / VID_FPS)
	END_TIME = START_TIME + datetime.timedelta(seconds=time_elapsed)


cap.release()


#write video data in /result/csv
storage.save_video_info(
    IS_CAM=IS_CAM,
    DATA_RECORD_FRAME=DATA_RECORD_FRAME,
    VID_FPS=VID_FPS,
    FRAME_SIZE=FRAME_SIZE,
    TRACK_MAX_AGE=TRACK_MAX_AGE,
    START_TIME=START_TIME,
    END_TIME=END_TIME
)

storage.close()


