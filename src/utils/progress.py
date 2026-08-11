import sys


def progress(frame_count):

    sys.stdout.write(
        f"\rProcessing frame: {frame_count}"
    )

    sys.stdout.flush()