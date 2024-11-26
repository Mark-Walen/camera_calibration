import cv2
import numpy
import os
import time
import threading
from queue import Queue

from core.camera_calibration.mono_calibrator import MonoCalibrator
from core.camera_calibration.stereo_calibrator import StereoCalibrator
from core.camera_calibration.calibrator import Patterns, CAMERA_MODEL
from core.logger import Logger

__all__ = [
    "stop_event",
    "is_running",

    "CalibrationNode",
    "OpenCVCalibrationNode"
]


__stop_event = threading.Event()

def stop_event():
    return __stop_event

def is_running():
    return not __stop_event.is_set()


class BufferQueue(Queue):
    """Slight modification of the standard Queue that discards the oldest item
    when adding an item and the queue is full.
    """

    def put(self, item, *args, **kwargs):
        # The base implementation, for reference:
        # https://github.com/python/cpython/blob/2.7/Lib/Queue.py#L107
        # https://github.com/python/cpython/blob/3.8/Lib/queue.py#L121
        with self.mutex:
            if 0 < self.maxsize == self._qsize():
                self._get()
            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()


class ConsumerThread(threading.Thread):
    def __init__(self, queue, function):
        threading.Thread.__init__(self)
        self.queue = queue
        self.function = function

    def run(self):
        try:
            while is_running():
                # Block until an item is available in the queue
                m = self.queue.get(block=True)
                if m is not None:
                    self.function(m)
        except Exception as e:
            import traceback
            traceback.print_exc()
            # print(f"ConsumerThread encountered an error: {e}")
        finally:
            stop_event().set()


class CalibrationNode:
    def __init__(self,
                 boards,
                 calibration_data_path,
                 cam_shot_path,
                 flags=0,
                 fisheye_flags=0,
                 pattern=Patterns.Chessboard,
                 camera_name='',
                 checkerboard_flags=0,
                 max_chessboard_speed=-1,
                 queue_size=1):

        self._logger = Logger("Camera Calibration")
        self._boards = boards
        self._calibration_data_path = calibration_data_path
        self._cam_shot_path = cam_shot_path
        self._calib_flags = flags
        self._fisheye_calib_flags = fisheye_flags
        self._checkerboard_flags = checkerboard_flags
        self._pattern = pattern
        self._camera_name = camera_name
        self._max_chessboard_speed = max_chessboard_speed

        self.q_mono = BufferQueue(queue_size)
        self.q_stereo = BufferQueue(queue_size)

        self.c = None

        self._last_display = None

        mth = ConsumerThread(self.q_mono, self.handle_monocular)
        mth.daemon = True
        mth.start()

        sth = ConsumerThread(self.q_stereo, self.handle_stereo)
        sth.daemon = True
        sth.start()

    def get_logger(self):
        return self._logger

    def redraw_stereo(self, *args):
        pass

    def redraw_monocular(self, *args):
        pass

    def queue_monocular(self, msg):
        self.q_mono.put(msg)

    def queue_stereo(self, lmsg, rmsg):
        self.q_stereo.put((lmsg, rmsg))

    def handle_monocular(self, msg):
        if self.c is None:
            if self._camera_name:
                self.c = MonoCalibrator(
                    self._boards, self._calibration_data_path, self._cam_shot_path, self._calib_flags, self._fisheye_calib_flags, self._pattern,
                    name=self._camera_name, checkerboard_flags=self._checkerboard_flags,
                    max_chessboard_speed=self._max_chessboard_speed)
            else:
                self.c = MonoCalibrator(
                    self._boards, self._calibration_data_path, self._cam_shot_path, self._calib_flags, self._fisheye_calib_flags, self._pattern,
                    checkerboard_flags=self._checkerboard_flags,
                    max_chessboard_speed=self._max_chessboard_speed)

        # This should just call the MonoCalibrator
        drawable = self.c.handle_msg(msg)
        self.displaywidth = drawable.scrib.shape[1]
        self.redraw_monocular(drawable)

    def handle_stereo(self, msg):
        if self.c is None:
            if self._camera_name:
                self.c = StereoCalibrator(
                    self._boards, self._calibration_data_path, self._cam_shot_path, self._calib_flags, self._fisheye_calib_flags, self._pattern,
                    name=self._camera_name, checkerboard_flags=self._checkerboard_flags,
                    max_chessboard_speed=self._max_chessboard_speed)
            else:
                self.c = StereoCalibrator(
                    self._boards, self._calibration_data_path, self._cam_shot_path, self._calib_flags, self._fisheye_calib_flags, self._pattern,
                    checkerboard_flags=self._checkerboard_flags,
                    max_chessboard_speed=self._max_chessboard_speed)

        drawable = self.c.handle_msg(msg)
        self.displaywidth = drawable.lscrib.shape[1] + drawable.rscrib.shape[1]
        self.redraw_stereo(drawable)

    def check_set_camera_info(self, response):
        if response.success:
            return True

        for i in range(10):
            print("!" * 80)
        print()
        print(f"Attempt to set camera info failed: {response.status_message}")
        print()
        for i in range(10):
            print("!" * 80)
        print()
        self.get_logger().error(f'Unable to set camera info for calibration. Failure message: {response.status_message}')
        return False

    def do_upload(self):
        self.c.report()
        print(self.c.ost())
        # info = self.c.as_message()


class OpenCVCalibrationNode(CalibrationNode):
    """ Calibration node with an OpenCV Gui """
    FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2

    def __init__(self, *args, **kwargs):

        CalibrationNode.__init__(self, *args, **kwargs)

        self.queue_display = BufferQueue(maxsize=1)
        self.initWindow()

    def spin(self):
        try:
            while is_running():
                if self.queue_display.qsize() > 0:
                    self.image = self.queue_display.get()
                    cv2.imshow("display", self.image)
                    k = cv2.waitKey(6) & 0xFF
                    if k in [27, ord('q')] or cv2.getWindowProperty("display", cv2.WND_PROP_VISIBLE) < 1:
                        break
                    elif k == ord('s') and self.image is not None:
                        self.screendump(self.image)
                else:
                    time.sleep(0.1)
        except Exception as e:
            self.get_logger().error(f'OpenCVCalibrationNode encountered an error: {e}')
        finally:
            stop_event().set()

    def initWindow(self):
        cv2.namedWindow("display", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("display", self.on_mouse)
        cv2.createTrackbar("Camera type: \n 0 : pinhole \n 1 : fisheye",
                           "display", 0, 1, self.on_model_change)
        cv2.createTrackbar("scale", "display", 0, 100, self.on_scale)

    @classmethod
    def putText(cls, img, text, org, color=(0, 0, 0)):
        cv2.putText(img, text, org, cls.FONT_FACE, cls.FONT_SCALE,
                    color, thickness=cls.FONT_THICKNESS)

    @classmethod
    def getTextSize(cls, text):
        return cv2.getTextSize(text, cls.FONT_FACE, cls.FONT_SCALE, cls.FONT_THICKNESS)[0]

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.displaywidth < x:
            if self.c.goodenough:
                if 180 <= y < 280:
                    print("**** Calibrating ****")
                    # Perform calibration in another thread to prevent UI blocking
                    threading.Thread(target=self.c.do_calibration,
                                     name="Calibration").start()
                    self.buttons(self._last_display)
                    self.queue_display.put(self._last_display)
            if self.c.calibrated:
                if 280 <= y < 380:
                    self.c.do_save()
                elif 380 <= y < 480:
                    # Only shut down if we set camera info correctly, #3993
                    if self.do_upload():
                        stop_event().set()

    def on_model_change(self, model_select_val):
        if self.c is None:
            print("Cannot change camera model until the first image has been received")
            return

        self.c.set_cammodel(
            CAMERA_MODEL.PINHOLE if model_select_val < 0.5 else CAMERA_MODEL.FISHEYE)

    def on_scale(self, scalevalue):
        if self.c and self.c.calibrated:
            self.c.set_alpha(scalevalue / 100.0)

    def button(self, dst, label, enable):
        dst.fill(255)
        size = (dst.shape[1], dst.shape[0])
        if enable:
            color = (155, 155, 80)
        else:
            color = (224, 224, 224)
        cv2.circle(dst, (size[0] // 2, size[1] // 2),
                   min(size) // 2, color, -1)
        (w, h) = self.getTextSize(label)
        self.putText(
            dst, label, ((size[0] - w) // 2, (size[1] + h) // 2), (255, 255, 255))

    def buttons(self, display):
        x = self.displaywidth
        self.button(display[180:280, x:x + 100], "CALIBRATE", self.c.goodenough)
        self.button(display[280:380, x:x + 100], "SAVE", self.c.calibrated)
        self.button(display[380:480, x:x + 100], "COMMIT", self.c.calibrated)

    def y(self, i):
        """Set up right-size images"""
        return 30 + 40 * i

    def screendump(self, im):
        i = 0
        _dump_png = f"{self._cam_shot_path}/dump0.png"
        while os.access(_dump_png, os.R_OK):
            i += 1
            _dump_png = f"{self._cam_shot_path}/dump{i}.png"
        cv2.imwrite(_dump_png, im)
        print(f"Saved screen dump to {_dump_png}")

    def redraw_monocular(self, drawable):
        height = drawable.scrib.shape[0]
        width = drawable.scrib.shape[1]

        display = numpy.zeros(
            (max(480, height), width + 100, 3), dtype=numpy.uint8)
        display[0:height, 0:width, :] = drawable.scrib
        display[0:height, width:width + 100, :].fill(255)

        self.buttons(display)
        if not self.c.calibrated:
            if drawable.params:
                for i, (label, lo, hi, progress) in enumerate(drawable.params):
                    (w, _) = self.getTextSize(label)
                    self.putText(display, label,
                                 (width + (100 - w) // 2, self.y(i)))
                    color = (0, 255, 0)
                    if progress < 1.0:
                        color = (0, int(progress * 255.), 255)
                    cv2.line(display,
                             (int(width + lo * 100), self.y(i) + 20),
                             (int(width + hi * 100), self.y(i) + 20),
                             color, 4)

        else:
            self.putText(display, "lin.", (width, self.y(0)))
            linerror = drawable.linear_error
            if linerror is None or linerror < 0:
                msg = "?"
            else:
                msg = "%.2f" % linerror
                # print "linear", linerror
            self.putText(display, msg, (width, self.y(1)))

        self._last_display = display
        self.queue_display.put(display)

    def redraw_stereo(self, drawable):
        height = drawable.lscrib.shape[0]
        width = drawable.lscrib.shape[1]

        display = numpy.zeros(
            (max(480, height), 2 * width + 100, 3), dtype=numpy.uint8)
        display[0:height, 0:width, :] = drawable.lscrib
        display[0:height, width:2 * width, :] = drawable.rscrib
        display[0:height, 2 * width:2 * width + 100, :].fill(255)

        self.buttons(display)

        if not self.c.calibrated:
            if drawable.params:
                for i, (label, lo, hi, progress) in enumerate(drawable.params):
                    (w, _) = self.getTextSize(label)
                    self.putText(display, label, (2 * width +
                                                  (100 - w) // 2, self.y(i)))
                    color = (0, 255, 0)
                    if progress < 1.0:
                        color = (0, int(progress * 255.), 255)
                    cv2.line(display,
                             (int(2 * width + lo * 100), self.y(i) + 20),
                             (int(2 * width + hi * 100), self.y(i) + 20),
                             color, 4)

        else:
            self.putText(display, "epi.", (2 * width, self.y(0)))
            if drawable.epierror == -1:
                msg = "?"
            else:
                msg = "%.2f" % drawable.epierror
            self.putText(display, msg, (2 * width, self.y(1)))
            # TODO dim is never set anywhere. Supposed to be observed chessboard size?
            if drawable.dim != -1:
                self.putText(display, "dim", (2 * width, self.y(2)))
                self.putText(display, "%.3f" %
                             drawable.dim, (2 * width, self.y(3)))

        self._last_display = display
        self.queue_display.put(display)
