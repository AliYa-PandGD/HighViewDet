from ultralytics import YOLO


class PoseEstimator:

    def __init__(
        self,
        model_path,
        device
    ):

        self.model = YOLO(model_path)
        self.device = device


    def estimate_batch(
        self,
        frame,
        tracks
    ):

        """
        Estimate pose for multiple tracked people.

        Parameters:
            frame:
                Original high-resolution frame

            tracks:
                List of confirmed Deep SORT tracks

        Returns:
            {
                track_id:
                {
                    "keypoints": ndarray,
                    "confidence": ndarray
                }
            }
        """


        batch_data = []


        frame_height, frame_width = frame.shape[:2]


        # Create person crops and store metadata together
        for track in tracks:


            x1, y1, x2, y2 = map(
                int,
                track.original_bbox
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


            batch_data.append(
                {
                    "track_id": track.track_id,
                    "crop": crop,
                    "offset": (x1, y1)
                }
            )


        if len(batch_data) == 0:
            return {}



        # Prepare batch for YOLO
        crops = [
            item["crop"]
            for item in batch_data
        ]


        # Batch inference
        results = self.model(
            crops,
            device=self.device,
            verbose=False
        )



        pose_results = {}



        # Match results with tracks
        for item, result in zip(
            batch_data,
            results
        ):


            if (
                result.keypoints is None
                or result.keypoints.xy.shape[0] == 0
            ):
                continue



            track_id = item["track_id"]


            x_offset, y_offset = item["offset"]



            keypoints = (
                result.keypoints.xy
                .cpu()
                .numpy()
            )


            # Convert crop coordinates
            # to original frame coordinates
            keypoints[:, :, 0] += x_offset
            keypoints[:, :, 1] += y_offset



            pose_results[track_id] = {

                "keypoints": keypoints,

                "confidence":
                    result.keypoints.conf.cpu().numpy()

            }


        return pose_results