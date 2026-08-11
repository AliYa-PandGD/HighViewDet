import csv
from datetime import datetime


class CSVStorage:

    def __init__(self, root_dir):

        self.result_directory = (
            root_dir /
            "results" /
            "csv"
        )

        self.result_directory.mkdir(
            parents=True,
            exist_ok=True
        )


        self.movement_file_path = (
            self.result_directory /
            "movement_data.csv"
        )

        self.crowd_file_path = (
            self.result_directory /
            "crowd_data.csv"
        )
        self.video_info_file_path = (
            self.result_directory /
            "video_info.csv"
        ) 


        self.movement_file = open(
            self.movement_file_path,
            'w',
            newline=''
        )

        self.crowd_file = open(
            self.crowd_file_path,
            'w',
            newline=''
        )
        self.video_info_file = open(
            self.video_info_file_path,
            'w',
            newline=''
        )


        self.movement_writer = csv.writer(
            self.movement_file
        )

        self.crowd_writer = csv.writer(
            self.crowd_file
        )
        self.video_info_writer = csv.writer(
            self.video_info_file
        )


        if self.movement_file_path.stat().st_size == 0:

            self.movement_writer.writerow(
                [
                    'Track ID',
                    'Entry time',
                    'Exit Time',
                    'Movement Tracks'
                ]
            )


        if self.crowd_file_path.stat().st_size == 0:

            self.crowd_writer.writerow(
                [
                    'Time',
                    'Human Count',
                    'Social Distance violate',
                    'Restricted Entry',
                    'Abnormal Activity'
                ]
            )
        if self.video_info_file_path.stat().st_size == 0:

            self.video_info_writer.writerow(
                [
                    'Is Camera',
                    'Data Record Frame',
                    'Video FPS',
                    'Frame Size',
                    'Track Max Age',
                    'Start Time',
                    'End Time'
                ]
            )

    def save_movement(self,track_id,entry_time,exit_time,movement_tracks):

        self.movement_writer.writerow(
            [
                track_id,
                entry_time,
                exit_time,
                movement_tracks
            ]
        )


    def save_crowd(self,time,human_count,social_distance_violate,restricted_entry,abnormal_activity):

        self.crowd_writer.writerow(
            [
                time,
                human_count,
                social_distance_violate,
                restricted_entry,
                abnormal_activity
            ]
        )

    def save_video_info(self, IS_CAM, DATA_RECORD_FRAME,VID_FPS,FRAME_SIZE,TRACK_MAX_AGE,START_TIME,END_TIME):

        if isinstance(START_TIME, float):
            START_TIME = datetime.fromtimestamp(
                START_TIME
            ).strftime("%d/%m/%Y, %H:%M:%S")

        if isinstance(END_TIME, float):
            END_TIME = datetime.fromtimestamp(
                END_TIME
            ).strftime("%d/%m/%Y, %H:%M:%S")


        self.video_info_writer.writerow(
            [
                IS_CAM,
                DATA_RECORD_FRAME,
                VID_FPS,
                FRAME_SIZE,
                TRACK_MAX_AGE,
                START_TIME,
                END_TIME
            ]
        )
        



    def close(self):
        self.movement_file.close()
        self.crowd_file.close()
        self.video_info_file.close()

    