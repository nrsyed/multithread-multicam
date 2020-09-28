import argparse
import multicam
import screeninfo


def get_parser():
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
    return parser


def process_args(multicam_args):
    """
    Args:
        multicam_args (argparse.Namespace): Parsed arguments returned by
            argparse.ArgumentParser.parse_args().
    Returns:
        A dict of keyword args that can be provided to multicam.show_videos().
    """
    # Convert to dict if not already.
    if isinstance(multicam_args, argparse.Namespace):
        multicam_args = vars(multicam_args)

    # Construct list of video stream device ids/paths.
    if multicam_args["all_cams"]:
        stream_ids = multicam.get_cam_ids()
    else:
        stream_ids = []
        for stream_id in multicam_args["stream_ids"]:
            if stream_id.isnumeric():
                # Assume this refers to an integer webcam device id.
                stream_id = int(stream_id)
            stream_ids.append(stream_id)

    grid_shape = "auto"
    if multicam_args["grid_shape"] is not None:
        grid_shape = multicam_args["grid_shape"]

    # Set window size options.
    win_flags = None
    win_size = "largest"

    if multicam_args["fit_screen"]:
        monitor = screeninfo.get_monitors()[0]
        win_size = (monitor.width, monitor.height)
    elif multicam_args["max_size"]:
        win_size = multicam_args["max_size"]
    elif multicam_args["resize"]:
        win_flags = cv2.WINDOW_NORMAL
    elif multicam_args["smallest"]:
        win_size = "smallest"

    return {
        "stream_ids": stream_ids,
        "grid_shape": grid_shape,
        "win_flags": win_flags,
        "win_size": win_size,
    }
