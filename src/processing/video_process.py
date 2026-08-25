import time
import datetime

import imutils
import cv2

from src.tracking.tracking import detect_human
from src.utils.colors import RGB_COLORS
from src.utils.progress import progress
from src.utils.pose_visualizer import draw_pose_batch




from src.config.setting import (
    SHOW_DETECT,
    DATA_RECORD,
    SHOW_TRACKING_ID,
    SHOW_PROCESSING_OUTPUT,
    VIDEO_CONFIG,
    DATA_RECORD_RATE
)


IS_CAM = VIDEO_CONFIG["IS_CAM"]



def _record_movement_data(storage, movement):

    storage.save_movement(
        track_id=movement.track_id,
        entry_time=movement.entry_time,
        exit_time=movement.exit_time,
        movement_tracks=movement.positions
    )



def _end_video(tracker, frame_count, storage):

    for track in tracker.tracks:

        if track.is_confirmed():

            track.exit = frame_count

            _record_movement_data(
                storage,
                track
            )



def video_process(cap,frame_size,encoder,tracker,storage,detector_obj,pose_estimator,body_orientation_estimator):

    t0 = None


    if IS_CAM:

        VID_FPS = None
        DATA_RECORD_FRAME = 1
        t0 = time.time()


    else:

        VID_FPS = cap.get(cv2.CAP_PROP_FPS)

        if VID_FPS <= 0:
            VID_FPS = 30

        DATA_RECORD_FRAME = int(
            VID_FPS / DATA_RECORD_RATE
        )



    def calculate_FPS():

        nonlocal VID_FPS

        if t0 is not None:

            t1 = time.time() - t0

            if t1 > 0:

                VID_FPS = frame_count / t1



    frame_count = 0
    display_frame_count = 0



    while True:


        ret, original_frame = cap.read()



        # Video finished
        if not ret:

            _end_video(
                tracker,
                frame_count,
                storage
            )

            if VID_FPS is None:

                calculate_FPS()

            break



        frame_count += 1



        # Avoid integer overflow
        if frame_count > 1000000:

            frame_count = 0
            display_frame_count = 0



        # Process only selected frames
        if frame_count % DATA_RECORD_FRAME != 0:

            continue



        display_frame_count += 1



        lower_res_frame = imutils.resize(
            original_frame,
            width=frame_size
        )



        current_datetime = datetime.datetime.now()



        if IS_CAM:

            record_time = current_datetime

        else:

            record_time = frame_count



        humans_detected, expired = detect_human(detector_obj,lower_res_frame,encoder,tracker,record_time)

        #Convert coordination of tracks on Low resolution frame to the original video resolution
        tracking_h, tracking_w = lower_res_frame.shape[:2]

        original_h, original_w = original_frame.shape[:2]


        scale_x = original_w / tracking_w
        scale_y = original_h / tracking_h

        for track in humans_detected:

            x1, y1, x2, y2 = map(
                int,
                track.to_tlbr().tolist()
            )


            original_bbox = [
                int(x1 * scale_x),
                int(y1 * scale_y),
                int(x2 * scale_x),
                int(y2 * scale_y)
            ]


            track.original_bbox = original_bbox




        #Estimate the pose of confirmed trackes
        pose_results = pose_estimator.estimate_batch(original_frame,humans_detected)

        # Attach pose and calculate body orientation
        for track in humans_detected:


            track.pose = None
            track.body_orientation = None


            if track.track_id not in pose_results:
                continue


            # Save pose
            track.pose = pose_results[
                track.track_id
            ]


            # Bounding box
            bbox = track.original_bbox


            # Estimate body orientation
            body_orientation = body_orientation_estimator.estimate(
                track.pose,
                bbox
            )

            if body_orientation is not None:
                #temporary debug
                print(
                    f"ID:{track.track_id} "
                    f"confidence:{body_orientation['pose_confidence']:.2f}"
                )
                track.body_orientation = body_orientation

            else:
                #temporary debug
                print(
                    f"ID:{track.track_id} no pose"
                )
                track.body_orientation = None


        #visualize pose detection
        frame = draw_pose_batch(original_frame,humans_detected)


        # Save finished tracks
        for movement in expired:

            _record_movement_data(
                storage,
                movement
            )



        # Draw detected people
        for track in humans_detected:


            x, y, w, h = track.original_bbox


            track_id = track.track_id



            if SHOW_DETECT:

                cv2.rectangle(
                    frame,
                    (x, y),
                    (w, h),
                    RGB_COLORS["green"],
                    2
                )



            if SHOW_TRACKING_ID:

                cv2.putText(
                    frame,
                    str(track_id),
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    RGB_COLORS["green"],
                    2
                )



        human_count = len(humans_detected)



        if SHOW_DETECT:

            cv2.putText(
                frame,
                f"Crowd count: {human_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                3
            )



        if DATA_RECORD:

            storage.save_crowd(
                time=record_time,
                human_count=human_count
            )



        if SHOW_PROCESSING_OUTPUT:

            display_frame = imutils.resize(
                frame,
                width=1280
            )

            cv2.imshow(
                "Processed Output",
                display_frame
            )

        else:

            progress(display_frame_count)



        if cv2.waitKey(1) & 0xFF == ord('q'):


            _end_video(
                tracker,
                frame_count,
                storage
            )


            if VID_FPS is None:

                calculate_FPS()


            break



    cv2.destroyAllWindows()


    return VID_FPS