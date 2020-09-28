import cv2
import multicam


def main():
    parser = multicam.get_parser()
    args = parser.parse_args()
    show_videos_kwargs = multicam.process_args(args)
    multicam.show_videos(**show_videos_kwargs)
