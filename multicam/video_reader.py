import abc
import queue
import threading

import cv2


class _VideoReader(abc.ABC):
    def __init__(self, src=0):
        """
        Class to read frames from a VideoCapture in a dedicated thread.

        Args:
            src (int|str): Video source. Int if webcam id, str if path to file
                or RTSP stream.
        """
        self.cap = cv2.VideoCapture(src)
        self.stopped = False

    def start(self):
        threading.Thread(target=self._get, args=()).start()
        return self

    @abc.abstractmethod
    def _get(self):
        """
        Method called in a thread to continually read frames from `self.cap`.
        """
        return

    def stop(self):
        self.stopped = True

    def __bool__(self):
        return not self.stopped

    def __del__(self):
        self.stop()
        self.cap.release()


class VideoReader(_VideoReader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grabbed, self.frame = self.cap.read()
        self.frames_read = 1

    def _get(self):
        while (not self.stopped) and self.grabbed:
            self.grabbed, self.frame = self.cap.read()
            self.frames_read += 1
        self.stop()


class VideoReaderQueue(_VideoReader):
    def __init__(self, *args, maxsize=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue = queue.Queue(maxsize)
        self.grabbed, frame = self.cap.read()
        self.frames_read = 1
        self.queue.put(frame)

    @property
    def frame(self):
        return self.queue.get()

    def _get(self):
        while (not self.stopped) and self.grabbed:
            self.grabbed, frame = self.cap.read()
            self.queue.put(frame)
            self.frames_read += 1
        self.stop()

    def __len__(self):
        return self.queue.qsize()
