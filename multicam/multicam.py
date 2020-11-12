from collections import deque
import math
import threading
import time

import cv2
import numpy as np

from .video_reader import VideoReader
from .video_shower import VideoShower


def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA):
    """
    TODO
    """
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
    stream_ids, grid_shape="auto", win_flags=None, win_size="largest",
    show_fps=False, out_frames=None, func=None, func_args=None, func_kwargs=None
):
    """
    Args:
        stream_ids (List[int|str]): A list of video stream paths or webcam 
            device ids.
        grid_shape (str|List[int], optional): A string or list of two ints
            representing the display grid shape.
        win_flags (int, optional): OpenCV imshow window properties flags.
        win_size (str|List[int], optional): Display window size.
        show_fps (bool, optional): Display number of frames processed per second.
        out_frames (list, optional): A list to which displayed frames will be
            appended for use by the caller.
        func (function, optional): If provided, frames read from each video
            stream will be given to `func` during each iteration of the main
            loop. `func` can process the frames as desired and must return
            a list of the processed frames.
        func_args (list, optional): Additional args for `func`.
        func_kwargs (dict, optional): Additional keyword args for `func`.
    """
    readers = [VideoReader(stream_id).start() for stream_id in stream_ids]
    shower = VideoShower(
        win_name="Video streams", win_flags=win_flags
    ).start()

    if func is not None:
        func_args = [] if func_args is None else func_args
        func_kwargs = {} if func_kwargs is None else func_kwargs

    # Number of frames to average for computing FPS.
    num_fps_frames = 30
    previous_fps = deque(maxlen=num_fps_frames)

    start_time = time.time()
    while shower and all(readers):
        loop_start_time = time.time()

        frames = [reader.frame for reader in readers]
        if func is not None:
            frames = func(frames, *func_args, **func_kwargs)

        # Place FPS in the corner of the first frame (if specified).
        if show_fps:
            cv2.putText(
                frames[0], f"{int(sum(previous_fps) / num_fps_frames)} fps",
                (2, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (255, 255, 255)
            )

        image_to_display = grid_stitch(
            frames, grid_shape=grid_shape, resize_to=win_size
        )

        if shower.stopped:
            break

        shower.frame = image_to_display
        if out_frames is not None:
            out_frames.append(image_to_display)

        previous_fps.append(int(1 / (time.time() - loop_start_time)))

    elapsed = time.time() - start_time

    for reader in readers:
        reader.stop()
    shower.stop()

    return elapsed
