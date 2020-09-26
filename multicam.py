import argparse
import glob
import re
import sys
import pdb

import cv2
import numpy as np
from screeninfo import get_monitors


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
    return cam_ids


def resize(img, width=None, height=None):
    h, w = img.shape[:2]

    if width and not height:
        new_w = width
        new_h = int((new_w / w) * h)
    elif height and not width:
        new_h = height
        new_w = int((new_h / h) * w)
    else:
        new_h = height
        new_w = width
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


def fit_screen(img, screen_w, screen_h):
    # Determine which way to resize img to avoid exceeding screen dims.
    img_h, img_w = img.shape[:2]

    resize_h_ratio = screen_h / img_h
    resize_h_new_w = int(resize_h_ratio * img_w)

    resize_w_ratio = screen_w / img_w
    resize_w_new_h = int(resize_w_ratio * img_h)

    resize_to_width = False
    if resize_h_new_w > screen_w:
        new_w = screen_w
        new_h = resize_w_new_h
        resize_to_width = True
    else:
        new_w = resize_h_new_w
        new_h = screen_h

    img = resize(img, width=new_w, height=new_h)

    bg = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    if resize_to_width:
        # Blank space above and below image.
        half_img_height = int(img.shape[0] / 2)
        bg_vctr = int(screen_h / 2)
        begin_v_img = bg_vctr - half_img_height
        bg[begin_v_img:(begin_v_img + img.shape[0]), :, :] = img
    else:
        # Blank space on left and right of image.
        half_img_width = int(img.shape[1] / 2)
        bg_hctr = int(screen_w / 2)
        begin_h_img = bg_hctr - half_img_width
        bg[:, begin_h_img:(begin_h_img + img.shape[1]), :] = img

    img = bg
    return img


def arrange_grid(*frames):
    pass


def stream(cam_ids=[0], screen_dims=None):
    caps = [cv2.VideoCapture(cam_id) for cam_id in cam_ids]
    rets = [cap.read() for cap in caps]

    cv2.namedWindow("stream", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("stream", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while all(ret[0] for ret in rets):
        rets = [cap.read() for cap in caps]
        frames = [ret[1] for ret in rets]
        disp = np.hstack(frames)

        if screen_dims is not None:
            screen_w, screen_h = screen_dims
            disp = fit_screen(disp, screen_w, screen_h)

        cv2.imshow("stream", disp)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    for cap in caps:
        cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cam_ids", nargs="*", default=[0])
    parser.add_argument("-a", "--all-cams", action="store_true")
    args = parser.parse_args()

    if args.all_cams:
        cam_ids = get_cam_ids()
    else:
        cam_ids = [int(cam_id) for cam_id in args.cam_ids]

    m = get_monitors()[0]
    screen_dims = (m.width, m.height)
    stream(cam_ids, screen_dims)
