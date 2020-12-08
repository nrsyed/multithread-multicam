import abc
import queue
import threading

import cv2


class _VideoReader(abc.ABC):
    def __init__(self, src=0):
        """
        Args:
            src (int|str): Video source. Int if webcam id, str if path to file
                or RTSP stream URI.
        """
        self.cap = cv2.VideoCapture(src)
        self.stopped = False

    def start(self):
        """
        Start reading from the VideoCapture in a separate thread.
        """
        threading.Thread(target=self._get, args=()).start()
        return self

    @abc.abstractmethod
    def _get(self):
        """
        Method called in a thread to continually read frames from `self.cap`.
        """
        return

    def stop(self):
        """
        Stop reading frames from the VideoCapture object.
        """
        self.stopped = True

    def __bool__(self):
        return not self.stopped

    def __del__(self):
        self.stop()
        self.cap.release()


class VideoReader(_VideoReader):
    """
    Class to read frames from a VideoCapture in a dedicated thread. This class
    does not add frames to a queue but instead, maintains a single `frame`
    attribute that is overwritten by the most recently read frame. If
    frame processing time exceeds the time required to read a frame, the
    intervening frames will be lost.

    Attributes:
        frame (np.ndarray): The current frame.

    .. note::
        The :class:`VideoReaderQueue` class should be used instead if each
        frame from the VideoCapture object is required.
    """
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
    """
    Class to read frames from a VideoCapture in a dedicated thread. Each frame
    is added to a queue as it is read, ensuring that all frames are made
    available to the caller.

    Attributes:
        frame (np.ndarray): The next frame in the queue.
    """
    def __init__(self, *args, maxsize=0, **kwargs):
        """
        Args:
            *args: Positional arguments to :meth:`_VideoReader.__init__`.
            maxsize (int): Queue capacity. If the queue is at max capacity,
                it will block subsequent calls to `VideoCapture.read` until
                items have been removed from the queue (by calling
                :meth:`frame`). If ``maxsize`` is `0`, capacity is infinite.
            **kwargs: Keyword arguments to :meth:`_VideoReader.__init__`.
        """
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
