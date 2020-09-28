import argparse
import os
import pathlib
import time

import torch

import multicam
import yolov3


def yolo_multicam_detect_demo(
    frames, net, device, class_names=None, duplicate_streams=0
):
    """
    Args:
        frames (List[np.ndarray]): List of frames from each video stream.
        net (yolov3.darknet.Darknet): Darknet class instance.
        device (str|torch.device): PyTorch device type.
        class_names (List[str]): List of Darknet model class names.
        duplicate_streams (int): Number of times to duplicate each frame.
            Each duplicate will be fed through the network like any other
            frame to simulate the processing of `duplicate` * `len(frames)`
            video streams.
    """
    frames = [frame for frame in frames]
    if duplicate_streams is not None and duplicate_streams > 0:
        frames *= duplicate_streams + 1

    results = yolov3.inference(net, frames, device=device)
    for i, frame in enumerate(frames):
        bbox_tlbr, class_prob, class_idx = results[i]
        yolov3.draw_boxes(
            frame, bbox_tlbr, class_idx=class_idx, class_names=class_names
        )
    return frames


if __name__ == "__main__":
    default_model = "yolov3-tiny"

    # Extend the default parser defined for the multicam package with
    # options relevant to the pytorch-yolov3 package and to this demo.
    parser = multicam.get_parser()
    parser.add_argument(
        "-c", "--config", type=pathlib.Path, default=f"{default_model}.cfg",
        metavar="<path>", help="Path to Darknet model .config file"
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cuda", metavar="<device>", 
        help="Device for inference ('cpu', 'cuda', 'cuda:0', etc.)"
    )
    parser.add_argument(
        "-n", "--class-names", type=pathlib.Path, default="coco.names", 
        metavar="<path>", help="Path to text file of class names"
    )
    parser.add_argument(
        "-w", "--weights", type=pathlib.Path,
        default=f"{default_model}.weights", metavar="<path>",
        help="Path to Darknet model weights file"
    )
    parser.add_argument(
        "-o", "--output", type=pathlib.Path, metavar="<path>",
        help="Path to output video file (.mp4 only)"
    )
    parser.add_argument(
        "--duplicate", type=int, metavar="<N>",
        help="Duplicate each video stream N times"
    )
    args = vars(parser.parse_args())

    if "cuda" in args["device"] and not torch.cuda.is_available():
        raise RuntimeError("CUDA selected but CUDA is not available")

    # Expand paths to absolute paths.
    path_args = ("config", "class_names", "weights", "output")
    for arg in path_args:
        if args[arg] is not None:
            args[arg] = str(args[arg].expanduser().absolute())

    class_names = None
    if args["class_names"] is not None:
        class_names = [
            line.strip() for line in open(args["class_names"], "r").readlines()
        ]

    # Instantiate Darknet class.
    net = yolov3.Darknet(args["config"], device=args["device"])
    net.load_weights(args["weights"])
    net.eval()
    net.cuda(device=args["device"])

    # Get keyword args for multicam.show_videos.
    show_videos_kwargs = multicam.process_args(args)

    out_frames = None
    if args["output"] is not None:
        out_frames = []

    duplicate_streams = None
    if args["duplicate"] is not None:
        duplicate_streams = args["duplicate"]

    # Wrap in try/except so that output video (if specified) is written even
    # if an exception occurs during execution.
    start_time = time.time()
    try:
        multicam.show_videos(
            **show_videos_kwargs, out_frames=out_frames,
            func=yolo_multicam_detect_demo, func_args=[net, args["device"]],
            func_kwargs={
                "class_names": class_names,
                "duplicate_streams": duplicate_streams
            }
        )
    except Exception as e:
        raise e
    finally:
        if args["output"] and out_frames:
            out_fps = 1 / ((time.time() - start_time) / len(out_frames))
            multicam.write_mp4(out_frames, out_fps, args["output"])
