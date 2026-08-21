import numpy as np


class BodyOrientationEstimator:

    def __init__(
        self,
        min_confidence=0.5
    ):

        self.min_confidence = min_confidence



    def estimate(self,pose,bbox):

        """
        Estimate body orientation from YOLO Pose keypoints.

        Body pose has a 180-degree ambiguity.
        Therefore, two possible front directions are returned.

        Returns:
        {
            "body_axis": vector,
            "front_candidates": [
                {
                    "direction": vector,
                    "confidence": float
                },
                {
                    "direction": vector,
                    "confidence": float
                }
            ],
            "pose_confidence": float
        }
        """


        if (
            pose is None
            or "keypoints" not in pose
            or pose["keypoints"] is None
        ):
            return None


        if len(pose["keypoints"]) == 0:
            return None


        keypoints = pose["keypoints"][0]


        if (
            "confidence" not in pose
            or pose["confidence"] is None
            or len(pose["confidence"]) == 0
        ):
            return None


        confidence = pose["confidence"][0]

        #temporary debug
        print(
            "keypoints:",
            keypoints.shape,
            "confidence:",
            confidence.shape
        )



        


        # COCO keypoints
        LEFT_SHOULDER = 5
        RIGHT_SHOULDER = 6
        LEFT_HIP = 11
        RIGHT_HIP = 12



        required_points = [
            LEFT_SHOULDER,
            RIGHT_SHOULDER,
            LEFT_HIP,
            RIGHT_HIP
        ]


        point_confidences = []


        for idx in required_points:

            if confidence[idx] < self.min_confidence:
                return None

            point_confidences.append(
                confidence[idx]
            )


        # YOLO keypoint reliability
        keypoint_confidence = np.mean(
            point_confidences
        )



        # -----------------------------
        # Calculate body lateral axis
        # -----------------------------

        left_shoulder = keypoints[LEFT_SHOULDER]
        right_shoulder = keypoints[RIGHT_SHOULDER]

        left_hip = keypoints[LEFT_HIP]
        right_hip = keypoints[RIGHT_HIP]



        shoulder_vector = (
            right_shoulder -
            left_shoulder
        )


        hip_vector = (
            right_hip -
            left_hip
        )



        body_axis = (
            shoulder_vector +
            hip_vector
        ) / 2



        body_width = np.linalg.norm(
            body_axis
        )


        if body_width == 0:
            return None



        body_axis = (
            body_axis /
            body_width
        )



        # -----------------------------
        # Body geometry confidence
        # -----------------------------

        x1, y1, x2, y2 = bbox

        bbox_width = x2 - x1


        if bbox_width <= 0:
            return None



        # Relative body width
        body_width_ratio = (
            body_width /
            bbox_width
        )


        geometry_confidence = np.clip(
            body_width_ratio / 0.35,
            0,
            1
        )



        pose_confidence = (
            0.7 * keypoint_confidence +
            0.3 * geometry_confidence
        )


        pose_confidence = float(
            np.clip(
                pose_confidence,
                0,
                1
            )
        )



        # -----------------------------
        # Two possible facing directions
        # -----------------------------

        front_1 = np.array(
            [
                -body_axis[1],
                 body_axis[0]
            ]
        )


        front_2 = -front_1


        # The confidence levels of front_1 and front_2 could be adjusted independently using body direction.
        # Body direction can be estimated from pose keypoints; however, this project also considers head direction.
        # To avoid unnecessary complexity and maintain consistency, the same confidence value is currently
        # assigned to both front_1 and front_2.
        return {


            "body_axis":
                body_axis,


            "front_candidates":

                [

                    {
                        "direction": front_1,
                        "confidence": pose_confidence
                    },


                    {
                        "direction": front_2,
                        "confidence": pose_confidence
                    }

                ],


            "pose_confidence":
                pose_confidence

        }
    