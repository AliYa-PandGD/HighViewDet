from ultralytics import YOLO
import numpy as np


class PoseEstimator:

    def __init__(
        self,
        model_path,
        device
    ):

        self.model = YOLO(model_path)
        self.device = device



    def estimate_batch(self,frame,tracks):

        """
        Estimate pose for multiple tracked people.

        Parameters:
            frame:
                Original image/frame

            tracks:
                List of Deep SORT Track objects

        Returns:
            Dictionary:
            {
                track_id:
                    {
                        "keypoints": ndarray,
                        "confidence": ndarray
                    }
            }
        """

        crops = []
        track_ids = []


        frame_height, frame_width = frame.shape[:2]


        # Create person crops
        for track in tracks:


            x1, y1, x2, y2 = map(
                int,
                track.to_tlbr()
            )


            # Keep coordinates inside image
            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(frame_width, x2)
            y2 = min(frame_height, y2)


            # Ignore invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue


            crop = frame[
                y1:y2,
                x1:x2
            ]


            crops.append(crop)
            track_ids.append(track.track_id)



        if len(crops) == 0:
            return {}



        # Batch inference
        results = self.model(crops,device=self.device,verbose=False)



        pose_results = {}


        # Match result with track ID
        for track_id, result in zip(track_ids,results):


            if result.keypoints is None:
                continue


            pose_results[track_id] = {

                "keypoints":
                    result.keypoints.xy.cpu().numpy(),

                "confidence":
                    result.keypoints.conf.cpu().numpy()

            }


        return pose_results