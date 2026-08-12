import datetime
from pathlib import Path

video_name = "Afternoon1.mp4"

# Video Path
ROOT_DIR = Path(__file__).resolve().parents[1]
video_path = (
    ROOT_DIR /
    "data" /
    "raw_video"/
    f"{video_name}"
)
VIDEO_CONFIG = {
	"VIDEO_CAP" : video_path,
	"IS_CAM" : False,
	"CAM_APPROX_FPS": 3,
	"HIGH_CAM": True,
	"START_TIME": datetime.datetime(2020, 11, 5, 0, 0, 0, 0)
}

# YOLO Model Configuration
# Supported versions: yolov8n/s/m/l/x, yolov9n/s/m/c/e,
#                     yolov26n/s/m 
# For GPU support, change device from 'cpu' to 0 (or GPU device ID)
YOLO_CONFIG = {
	"MODEL_VERSION": "yolov8n.pt",  # Change to yolov26, yolov9, etc.
	"DEVICE": "auto"              # Use "cpu" or GPU device ID (0, 1, etc.)
}

# Show individuals detected
SHOW_PROCESSING_OUTPUT = True
# Show individuals detected
SHOW_DETECT = True
# Data record
DATA_RECORD = True
# Data record rate (data record per frame)
DATA_RECORD_RATE = 5
# Check for restricted entry
RE_CHECK = False
# Restricted entry time (H:M:S)
RE_START_TIME = datetime.time(0,0,0) 
RE_END_TIME = datetime.time(23,0,0)
# Check for social distance violation
SD_CHECK = False
# Show tracking id
SHOW_TRACKING_ID = True
# Threshold for human detection minumun confindence
MIN_CONF = 0.2
# Threshold for Non-maxima surpression
NMS_THRESH = 0.2
# Resize frame for processing
FRAME_SIZE = 1080
# Tracker max missing age before removing (seconds)
TRACK_MAX_AGE = 3