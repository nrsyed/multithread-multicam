import glob
import re

import cv2
import screeninfo

import multicam


def get_cam_ids():
    expr = r"/dev/video([0-9]+)"
    dev_ids = []
    for dev in glob.glob("/dev/video*"):
        match = re.match(expr, dev)
        if match is not None:
            dev_ids.append(int(match.groups()[0]))

    cam_ids = []
    for dev_id in dev_ids:
        cap = cv2.VideoCapture(dev_id)
        if cap.isOpened():
            cam_ids.append(dev_id)
        cap.release()
    return sorted(cam_ids)


def write_mp4(frames, fps, fpath):
    """
    Write frames to an .mp4 video.

    Args:
        frames (List[np.ndarray]): List of frames to be written.
        fps (int): Framerate of the output video.
        fpath (str): Path to output video file.
    """
    if not fpath.endswith(".mp4"):
        fpath += ".mp4"

    h, w = frames[0].shape[:2]

    writer = cv2.VideoWriter(
        fpath, cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (w, h)
    )

    for frame in frames:
        writer.write(frame)
    writer.release()
