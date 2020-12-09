import threading

import cv2


class VideoShower():
    def __init__(self, frame=None, win_name="Video", win_flags=None):
        """
        Class to show frames in a dedicated thread.

        Args:
            frame (np.ndarray): Initial frame to display.
            win_name (str): Name of `cv2.imshow` window.
            win_flags (int): Options for `cv2.namedWindow`.
        """
        self.frame = frame
        self.win_name = win_name
        self.win_flags = win_flags
        self.stopped = False

    def start(self):
        """
        Start showing frames in a separate thread.
        """
        threading.Thread(target=self._show, args=()).start()
        return self

    def _show(self):
        """
        Method called within thread to show new frames.
        """
        cv2.namedWindow(self.win_name, self.win_flags)
        while not self.stopped:
            # Only calling imshow when a new frame is set improves performance.
            if self.frame is not None:
                cv2.imshow(self.win_name, self.frame)
                self.frame = None

            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        """
        Stop showing frames and destroy the OpenCV window.
        """
        cv2.destroyWindow(self.win_name)
        self.stopped = True
