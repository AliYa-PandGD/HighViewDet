from ultralytics import YOLO

# implementing YOLO pose detection
class PoseEstimator:

    def __init__(self, model_path, device):

        self.model = YOLO(model_path)

        self.device = device


    def estimate_batch(self, frame, humans_detected):
        """
        x1, y1, x2, y2 = map(float, bbox)

        #control if a person is close to the edge
        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return None

        person_crop = frame[y1:y2,x1:x2]

        results = self.model(person_crop,device=self.device,verbose=False)

        if result.keypoints is None:
            return None
        
        return result
        """

        return None