import cv2
import numpy as np


# COCO 17 keypoint skeleton
POSE_CONNECTIONS = [

    # Head
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),

    # Arms
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),

    # Torso
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),

    # Legs
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16)
]


def draw_pose_batch(
    frame,
    tracks
):
    """
    Draw pose and body orientation for multiple tracks.

    Parameters
    ----------
    frame:
        Current video frame

    tracks:
        List of Deep SORT Track objects

    Returns
    -------
    frame
        Annotated frame
    """

    for track in tracks:

        frame = draw_single_pose(
            frame,
            track
        )


    return frame



def draw_single_pose(
    frame,
    track
):
    """
    Draw pose information for one track.
    """


    # No pose available
    if (
        not hasattr(track, "pose")
        or track.pose is None
    ):
        return frame



    keypoints_data = track.pose.get("keypoints")

    if (
        keypoints_data is None
        or len(keypoints_data) == 0
    ):
        return frame


    keypoints = keypoints_data[0]

    if keypoints.shape[0] != 17:
        return frame



    # -----------------------------
    # Draw keypoints
    # -----------------------------

    for idx, point in enumerate(keypoints):

        x, y = map(
            int,
            point
        )


        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 255, 255),
            -1
        )



    # -----------------------------
    # Draw skeleton
    # -----------------------------

    for start, end in POSE_CONNECTIONS:


        x1, y1 = map(
            int,
            keypoints[start]
        )

        x2, y2 = map(
            int,
            keypoints[end]
        )


        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )



    # -----------------------------
    # Draw body orientation
    # -----------------------------

    if (
        not hasattr(track, "body_orientation")
        or track.body_orientation is None
    ):
        return frame



    orientation = track.body_orientation



    # Center of torso
    center = np.mean(
        [
            keypoints[5],   # left shoulder
            keypoints[6],   # right shoulder
            keypoints[11],  # left hip
            keypoints[12]   # right hip
        ],
        axis=0
    )


    center = center.astype(int)



    # -----------------------------
    # Body axis
    # -----------------------------

    body_axis = orientation["body_axis"]


    body_end = (
        center +
        body_axis * 80
    ).astype(int)


    cv2.arrowedLine(
        frame,
        tuple(center),
        tuple(body_end),
        (255, 0, 0),
        3
    )



    # -----------------------------
    # Front candidates
    # -----------------------------

    colors = [
        (0, 0, 255),       # red
        (255, 0, 255)      # purple
    ]


    for candidate, color in zip(
        orientation["front_candidates"],
        colors
    ):


        direction = candidate["direction"]


        end = (
            center +
            direction * 100
        ).astype(int)



        cv2.arrowedLine(
            frame,
            tuple(center),
            tuple(end),
            color,
            3
        )



        confidence = candidate["confidence"]



        cv2.putText(
            frame,
            f"{confidence:.2f}",
            tuple(end),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )



    # -----------------------------
    # Track information
    # -----------------------------

    



    cv2.putText(
        frame,
        f"Pose:{orientation['pose_confidence']:.2f}",
        (
            center[0],
            center[1] + 20
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255,255,255),
        2
    )


    return frame