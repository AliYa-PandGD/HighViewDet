from ultralytics import YOLO

# implementing YOLO pose detection
class PoseEstimator:

    def __init__(self, model_path, device):

        self.model = YOLO(model_path)

        self.device = device


    def estimate(self, frame, bbox):

        x1, y1, x2, y2 = bbox

        crop = frame[
            y1:y2,
            x1:x2
        ]

        results = self.model(
            crop,
            device=self.device,
            verbose=False
        )

        return results