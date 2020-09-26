import argparse
from collections import deque
import glob
import math
import re
import threading
import time

import cv2
import numpy as np

try:
    import screeninfo
except ModuleNotFoundError:
    pass


class VideoGetter():
    def __init__(self, src=0):
        """
        Class to read frames from a VideoCapture in a dedicated thread.

        Args:
            src (int|str): Video source. Int if webcam id, str if path to file.
        """
        self.cap = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.get, args=()).start()
        return self

    def get(self):
        """
        Method called in a thread to continually read frames from `self.cap`.
        This way, a frame is always ready to be read. Frames are not queued;
        if a frame is not read before `get()` reads a new frame, previous
        frame is overwritten.
        """
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                self.grabbed, self.frame = self.cap.read()

    def stop(self):
        self.stopped = True


class VideoShower():
    def __init__(self, frame=None, win_name="Video", win_flags=None):
        """
        Class to show frames in a dedicated thread.

        Args:
            frame (np.ndarray): (Initial) frame to display.
            win_name (str): Name of `cv2.imshow()` window.
        """
        self.frame = frame
        self.win_name = win_name
        self.win_flags = win_flags
        self.stopped = False

    def start(self):
        threading.Thread(target=self.show, args=()).start()
        return self

    def show(self):
        """
        Method called within thread to show new frames.
        """
        cv2.namedWindow(self.win_name, self.win_flags)
        while not self.stopped:
            # We can actually see an ~8% increase in FPS by only calling
            # cv2.imshow when a new frame is set with an if statement. Thus,
            # set `self.frame` to None after each call to `cv2.imshow()`.
            if self.frame is not None:
                cv2.imshow(self.win_name, self.frame)
                self.frame = None

            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        cv2.destroyWindow(self.win_name)
        self.stopped = True


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


def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]

    if width and height:
        if w == width and h == height:
            return img
        else:
            new_h = height
            new_w = width
    if width and not height:
        new_w = width
        new_h = int((new_w / w) * h)
    elif height and not width:
        new_h = height
        new_w = int((new_h / h) * w)
    img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    return img


def make_square(src, bg_fill_color=(0, 0, 0)):
    h, w = src.shape[:2]
    bg_fill_color = np.array(bg_fill_color, dtype=src.dtype)

    if h == w:
        return src
    elif h > w:
        dst = bg_fill_color * np.ones((h, h, 3), dtype=src.dtype)
        start_col_idx = (h - w) // 2
        dst[:, start_col_idx:(start_col_idx + w), :] = src
    elif w > h:
        dst = bg_fill_color * np.ones((w, w, 3), dtype=src.dtype)
        start_row_idx = (w - h) // 2
        dst[start_row_idx:(start_row_idx + h), :, :] = src
    return dst


def grid_stitch(
    images, grid_shape="auto", resize_to="largest", grid_fill_color=(50, 50, 50)
):
    """
    Args:
        images (List[np.ndarray]): List of images with which to populate grid.
        grid_shape (str|Tup[int]): Grid shape (rows, cols), or "auto".
        resize_to (str|Tup[int]):
            "largest": Resize images to match largest image (perimeter).
            "smallest": Resize images to smallest image (perimeter).
            (int, int): Resize images to fit within the specified maximum
                grid size (w, h).
    """
    # TODO: add padding option?

    if grid_shape == "auto":
        n = 1
        while len(images) > n**2:
            n += 1

        dim1 = n
        dim2 = math.ceil(len(images) / dim1)
        
        grid_rows = dim2
        grid_cols = dim1
    else:
        grid_rows, grid_cols = grid_shape

    images = [make_square(image) for image in images]

    # Sort dimensions of each image by area in descending order.
    sorted_dims = sorted(
        [image.shape[:2] for image in images],
        key=lambda dims: dims[0] * dims[1],
        reverse=True
    )

    if isinstance(resize_to, str):
        if resize_to == "largest":
            dims = sorted_dims[0]
        elif resize_to == "smallest":
            dims = sorted_dims[-1]
        else:
            raise ValueError("Invalid str option for resize_to")

        images = [
            resize(image, width=dims[1], height=dims[0]) for image in images
        ]
    elif isinstance(resize_to, (list, tuple)):
        max_w, max_h = resize_to

        # Transpose grid shape if max grid size is larger vertically but
        # grid shape is larger horizontally.
        if max_h > max_w and grid_cols > grid_rows:
            grid_cols, grid_rows = grid_rows, grid_cols

        # Compute image dimensions based on max grid size in either direction
        # (images are assumed to be square at this point).
        image_dim = math.floor(max_h / grid_rows)
        if grid_cols > grid_rows:
            _image_dim = math.floor(max_w / grid_cols)
            if _image_dim * grid_rows <= max_h:
                image_dim = _image_dim

        dims = [image_dim, image_dim]
        images = [
            resize(image, width=image_dim, height=image_dim) for image in images
        ]

    grid_dims = [grid_cols * dims[0], grid_rows * dims[1]]
    dst = (
        np.array(grid_fill_color, dtype=images[0].dtype)
        * np.ones((grid_dims[1], grid_dims[0], 3), dtype=images[0].dtype)
    )

    # Populate grid across rows first.
    for i, image in enumerate(images):
        r = i // grid_cols
        c = i - (r * grid_cols)

        # All image dimensions should match `dims` at this point.
        ul_x = c * dims[0]
        ul_y = r * dims[1]

        dst[ul_y:(ul_y + dims[1]), ul_x:(ul_x + dims[0]), :] = image

    return dst
        

def show_videos(
    cam_ids, grid_shape="auto", win_flags=None, win_size="largest",
    show_fps=False, func=None, func_args=None, func_kwargs=None
):
    """
    TODO
    """
    getters = [VideoGetter(cam_id).start() for cam_id in cam_ids]
    shower = VideoShower(
        win_name="Video streams", win_flags=win_flags
    ).start()

    if func is not None:
        func_args = [] if func_args is None else func_args
        func_kwargs = {} if func_kwargs is None else func_kwargs

    # Number of frames to average for computing FPS.
    num_fps_frames = 30
    previous_fps = deque(maxlen=num_fps_frames)

    while True:
        loop_start_time = time.time()

        if shower.stopped or any(getter.stopped for getter in getters):
            break

        frames = [getter.frame for getter in getters]
        if func is not None:
            frames = func(frames, *func_args, **func_kwargs)

        image_to_display = grid_stitch(
            frames, grid_shape=grid_shape, resize_to=win_size
        )

        if show_fps:
            cv2.putText(
                image_to_display,
                f"{int(sum(previous_fps) / num_fps_frames)} fps",
                (2, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (255, 255, 255)
            )

        shower.frame = image_to_display
        previous_fps.append(int(1 / (time.time() - loop_start_time)))

    shower.stop()
    for getter in getters:
        getter.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stream_ids", nargs="*", default=[0],
        help="Video stream ids, i.e., webcam device integer or stream URL/path"
        " (default 0)"
    )
    parser.add_argument("-a", "--all-cams", action="store_true")
    parser.add_argument(
        "-g", "--grid-shape", nargs=2, type=int, metavar=("<rows>", "<cols>"),
        help="Grid shape (one video stream is displayed per grid cell); "
        "if omitted, optimal grid shape is determined automatically"
    )

    win_size_group = parser.add_argument_group(
        "display window options",
        "only 1 option from this list may be selected (default \"largest\")"
    )
    win_size_args = win_size_group.add_mutually_exclusive_group()
    win_size_args.add_argument(
        "-f", "--fit-screen", action="store_true",
        help="Size display window to fill available screen space"
    )
    win_size_args.add_argument(
        "-l", "--largest", action="store_true",
        help="Resize all video streams to dimensions of largest stream "
        "(default); aspect ratio is preserved"
    )
    win_size_args.add_argument(
        "-m", "--max-size", nargs=2, type=int, metavar=("<width>", "<height>"),
        help="Fit display window within maximum specified (width, height)"
    )
    win_size_args.add_argument(
        "-r", "--resize", action="store_true",
        help="Allow display window to be resizeable by user"
    )
    win_size_args.add_argument(
        "-s", "--smallest", action="store_true",
        help="Resize all video streams to dimensions of smallest stream; "
        "aspect ratio is preserved"
    )
    args = parser.parse_args()

    # Construct list of video stream device ids/paths.
    if args.all_cams:
        stream_ids = get_cam_ids()
    else:
        stream_ids = []
        for stream_id in args.stream_ids:
            if stream_id.isnumeric():
                # Assume this refers to an integer webcam device id.
                stream_id = int(stream_id)
            stream_ids.append(stream_id)

    grid_shape = "auto"
    if args.grid_shape is not None:
        grid_shape = args.grid_shape

    # Set window size options.
    win_flags = None
    win_size = "largest"

    if args.fit_screen:
        monitor = screeninfo.get_monitors()[0]
        win_size = (monitor.width, monitor.height)
    elif args.max_size:
        win_size = args.max_size
    elif args.resize:
        win_flags = cv2.WINDOW_NORMAL
    elif args.smallest:
        win_size = "smallest"

    show_videos(
        stream_ids, grid_shape=grid_shape, win_flags=win_flags,
        win_size=win_size
    )
