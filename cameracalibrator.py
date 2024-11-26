#!/usr/bin/env python
#
# Software License Agreement (BSD License)
#
# Copyright (c) 2009, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of the Willow Garage nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
import os
import sys
import threading
import time
from typing import List

import cv2
import tempfile

from core.camera_calibration.calibrator import ChessboardInfo, Patterns
from core.camera_calibration.camera_calibrator import OpenCVCalibrationNode, stop_event, is_running
from core.logger import Logger

logger = Logger("cameracalibrator")


class CaptureStereo(threading.Thread):
    def __init__(self, source: List[int], queue_func=None, height=0, width=0):
        threading.Thread.__init__(self)
        if source is None:
            source = [0]
            logger.warning("No video sources provided. Use default video source 0.")

        self.cap0 = cv2.VideoCapture(source[0])
        self.cap1 = cv2.VideoCapture(source[1]) if len(source) > 1 else None

        self.queue_func = queue_func
        self.height = height
        self.width = width
        self.daemon = True

        if not self.cap0.isOpened():
            logger.error("[Error]: Could not open video stream {}.".format(source[0]))
            stop_event().set()
            return

        if self.cap1 and not self.cap1.isOpened():
            logger.error("[Error]: Could not open video stream {}.".format(source[1]))
            stop_event().set()
            return

        # Set resolution based on inputs
        if self.cap1:   # Stereo cameras
            if width > 0 and height > 0:
                self.cap0.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

                self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        else:           # Single camera (stereo in one frame)
            if width > 0 and height > 0:
                self.cap0.set(cv2.CAP_PROP_FRAME_WIDTH, self.width * 2)
                self.cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            else:
                self.height, self.width = 480, 640
                self.cap0.set(cv2.CAP_PROP_FRAME_WIDTH, self.width * 2)
                self.cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        logger.debug(
            f"Camera 0 resolution: {self.cap0.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap0.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        if self.cap1:
            logger.debug(
                f"Camera 1 resolution: {self.cap1.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

    def run(self):
        try:
            while is_running():
                ret, frame = self.cap0.read()
                if not ret:
                    break

                # Split the frame into left and right images
                lframe = frame[:, :self.width]  # Left frame (left half of the image)
                if self.cap1:
                   ret, rframe = self.cap1.read()
                   if not ret:
                       break
                else:
                    rframe = frame[:, self.width:]  # Right frame (right half of the image)

                if self.queue_func is not None:
                    self.queue_func(lframe, rframe)

                time.sleep(0.01)
            logger.info("Capture thread terminated.")
        except Exception as e:
            logger.error("capture_stereo: exception {}".format(e))
        finally:
            self.cap0.release()
            if self.cap1:
                self.cap1.release()
            stop_event().set()


class CaptureMono(threading.Thread):
    def __init__(self, source: int, queue_func=None, height=0, width=0):
        threading.Thread.__init__(self)

        self.cap = cv2.VideoCapture(source)
        self.queue_func = queue_func
        self.height = height
        self.width = width
        self.daemon = True

        if not self.cap.isOpened():
            logger.error("Error: Could not open video stream.")
            stop_event().set()
            return

        if width >= 0 and height >= 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def run(self):
        try:
            while is_running():
                ret, frame = self.cap.read()
                if not ret:
                    break

                if self.queue_func is not None:
                    self.queue_func(frame)

                time.sleep(0.01)
            logger.info("Capture thread terminated.")
        except Exception as e:
            logger.error("capture_stereo: exception {}".format(e))
        finally:
            self.cap.release()
            stop_event().set()

# Function to validate a path and fall back to default if not valid
def get_valid_path(provided_path, default_path, label):
    if provided_path and os.path.isdir(provided_path):
        return provided_path
    elif provided_path:
        logger.warning(f"The provided {label} directory '{provided_path}' does not exist. "
              f"Using default directory: '{default_path}'.")
    return default_path


def optionsValidCharuco(options, parser):
    """
    Validates the provided options when the pattern type is 'charuco'
    """
    if options.pattern != 'charuco':
        return False

    n_boards = len(options.size)
    if (n_boards != len(options.square) or n_boards != len(options.charuco_marker_size) or n_boards !=
            len(options.aruco_dict)):
        parser.error(
            "When using ChArUco boards, --size, --square, --charuco_marker_size, and --aruco_dict "
            + "must be specified for each board")
        return False

    # TODO: check for fisheye and stereo (not implemented with ChArUco)
    return True


def parse_args(argv):
    from optparse import OptionParser, OptionGroup
    parser = OptionParser("%prog <mono|stereo> --size SIZE1 --square SQUARE1 [ --size SIZE2 --square SQUARE2 ]",
                          description="Camera calibration utility for mono or stereo setups.")
    parser.add_option("-S", "--sources", default=[0], action="append", help="Video sources to calibrate.")
    parser.add_option("-w", "--width", type="int", default=640,
                      help="Each video frame width (default: 640).")
    parser.add_option("-H", "--height", type="int", default=480,
                      help="Each video frame height (default: 480).")

    parser.add_option("-c", "--camera_name",
                      type="string", default='narrow_stereo',
                      help="name of the camera to appear in the calibration file")
    group = OptionGroup(
        parser, "Chessboard Options",
        "You must specify one or more chessboards as pairs of --size and --square options.")
    group.add_option("-p", "--pattern",
                     type="string", default="chessboard",
                     help="calibration pattern to detect - 'chessboard', 'circles', 'acircles', 'charuco'\n" +
                          "  if 'charuco' is used, a --charuco_marker_size and --aruco_dict argument must be supplied\n" +
                          "  with each --size and --square argument")
    group.add_option(
        "-s", "--size", action="append", default=[],
        help="chessboard size as NxM, counting interior corners (e.g. a standard chessboard is 7x7)")
    group.add_option("-q", "--square",
                     action="append", default=[],
                     help="chessboard square size in meters")
    group.add_option("-m", "--charuco_marker_size",
                     action="append", default=[],
                     help="ArUco marker size (meters); only valid with `-p charuco`")
    group.add_option("-d", "--aruco_dict",
                     action="append", default=[],
                     help="ArUco marker dictionary; only valid with `-p charuco`; one of 'aruco_orig', '4x4_250', " +
                          "'5x5_250', '6x6_250', '7x7_250'")
    parser.add_option_group(group)
    group = OptionGroup(parser, "ROS Communication Options")
    # group.add_option(
    #     "--approximate", type="float", default=0.0,
    #     help="allow specified slop (in seconds) when pairing images from unsynchronized stereo cameras")
    group.add_option("--queue-size",
                     type="int", default=1,
                     help="image queue size (default %default, set to 0 for unlimited)")
    parser.add_option_group(group)
    group = OptionGroup(parser, "Calibration Optimizer Options")
    group.add_option("--fix-principal-point",
                     action="store_true", default=False,
                     help="for pinhole, fix the principal point at the image center")
    group.add_option("--fix-aspect-ratio",
                     action="store_true", default=False,
                     help="for pinhole, enforce focal lengths (fx, fy) are equal")
    group.add_option("--zero-tangent-dist",
                     action="store_true", default=False,
                     help="for pinhole, set tangential distortion coefficients (p1, p2) to zero")
    group.add_option(
        "-k", "--k-coefficients", type="int", default=2, metavar="NUM_COEFFS",
        help="for pinhole, number of radial distortion coefficients to use (up to 6, default %default)")

    group.add_option(
        "--fisheye-recompute-extrinsicsts", action="store_true", default=False,
        help="for fisheye, extrinsic will be recomputed after each iteration of intrinsic optimization")
    group.add_option("--fisheye-fix-skew",
                     action="store_true", default=False,
                     help="for fisheye, skew coefficient (alpha) is set to zero and stay zero")
    group.add_option("--fisheye-fix-principal-point",
                     action="store_true", default=False,
                     help="for fisheye,fix the principal point at the image center")
    group.add_option(
        "--fisheye-k-coefficients", type="int", default=4, metavar="NUM_COEFFS",
        help="for fisheye, number of radial distortion coefficients to use fixing to zero the rest (up to 4, default %default)")

    group.add_option("--fisheye-check-conditions",
                     action="store_true", default=False,
                     help="for fisheye, the functions will check validity of condition number")

    group.add_option("--disable_calib_cb_fast_check", action='store_true', default=False,
                     help="uses the CALIB_CB_FAST_CHECK flag for findChessboardCorners")
    group.add_option("--max-chessboard-speed", type="float", default=-1.0,
                     help="Do not use samples where the calibration pattern is moving faster \
                         than this speed in px/frame. Set to eg. 0.5 for rolling shutter cameras.")
    parser.add_option_group(group)

    group = OptionGroup(parser, "Calibration Data Path Options")
    group.add_option("--calibration-data-path", type= "string", default="",
                     help="Path to calibration data directory. \
                     Calibration data includes camera parameters and sampled image use to calibrate. \
                     Default path is system temp directory.")
    group.add_option("--cam-shot-path", type="string", default="",
                     help="Path to cam shot image directory (Triggered by hotkey 's')."
                         "Cam shot images will be use to calibration too."
                         "Default path is system temp directory.")

    parser.add_option_group(group)
    return parser, parser.parse_args(argv)


def main():
    args = sys.argv
    if len(args) < 2:
        logger.error("Mode ('mono' or 'stereo') must be specified.")
        return

    mode = args[1]
    parser, (options, _) = parse_args(args[2:])

    if len(options.size) != len(options.square):
        parser.error("Number of size and square inputs must be the same!")

    if not options.square:
        options.square.append("0.108")
        options.size.append("8x6")

    boards = []
    if options.pattern == "charuco" and optionsValidCharuco(options, parser):
        for (sz, sq, ms, ad) in zip(options.size, options.square, options.charuco_marker_size, options.aruco_dict):
            size = tuple([int(c) for c in sz.split('x')])
            boards.append(ChessboardInfo(
                'charuco', size[0], size[1], float(sq), float(ms), ad))
    else:
        for (sz, sq) in zip(options.size, options.square):
            size = tuple([int(c) for c in sz.split('x')])
            boards.append(ChessboardInfo(
                options.pattern, size[0], size[1], float(sq)))

    # Pinhole opencv calibration options parsing
    num_ks = options.k_coefficients

    calib_flags = 0
    if options.fix_principal_point:
        calib_flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    if options.fix_aspect_ratio:
        calib_flags |= cv2.CALIB_FIX_ASPECT_RATIO
    if options.zero_tangent_dist:
        calib_flags |= cv2.CALIB_ZERO_TANGENT_DIST
    if num_ks > 3:
        calib_flags |= cv2.CALIB_RATIONAL_MODEL
    if num_ks < 6:
        calib_flags |= cv2.CALIB_FIX_K6
    if num_ks < 5:
        calib_flags |= cv2.CALIB_FIX_K5
    if num_ks < 4:
        calib_flags |= cv2.CALIB_FIX_K4
    if num_ks < 3:
        calib_flags |= cv2.CALIB_FIX_K3
    if num_ks < 2:
        calib_flags |= cv2.CALIB_FIX_K2
    if num_ks < 1:
        calib_flags |= cv2.CALIB_FIX_K1

    # Opencv calibration flags parsing:
    num_ks = options.fisheye_k_coefficients
    fisheye_calib_flags = 0
    if options.fisheye_fix_principal_point:
        fisheye_calib_flags |= cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT
    if options.fisheye_fix_skew:
        fisheye_calib_flags |= cv2.fisheye.CALIB_FIX_SKEW
    if options.fisheye_recompute_extrinsicsts:
        fisheye_calib_flags |= cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    if options.fisheye_check_conditions:
        fisheye_calib_flags |= cv2.fisheye.CALIB_CHECK_COND
    if num_ks < 4:
        fisheye_calib_flags |= cv2.fisheye.CALIB_FIX_K4
    if num_ks < 3:
        fisheye_calib_flags |= cv2.fisheye.CALIB_FIX_K3
    if num_ks < 2:
        fisheye_calib_flags |= cv2.fisheye.CALIB_FIX_K2
    if num_ks < 1:
        fisheye_calib_flags |= cv2.fisheye.CALIB_FIX_K1

    pattern = Patterns.Chessboard
    if options.pattern == 'circles':
        pattern = Patterns.Circles
    elif options.pattern == 'acircles':
        pattern = Patterns.ACircles
    elif options.pattern == 'charuco':
        pattern = Patterns.ChArUco
    elif options.pattern != 'chessboard':
        print('Unrecognized pattern %s, defaulting to chessboard' %
              options.pattern)

    if options.disable_calib_cb_fast_check:
        checkerboard_flags = 0
    else:
        checkerboard_flags = cv2.CALIB_CB_FAST_CHECK

    # System default temp directory
    default_temp_dir = tempfile.gettempdir()
    # Resolve paths
    calibration_data_path = get_valid_path(options.calibration_data_path, default_temp_dir, "calibration data")
    cam_shot_path = get_valid_path(options.cam_shot_path, default_temp_dir, "cam shot image")

    if mode == "-h" or mode == "--help":
        parser.print_help()
        exit(0)

    try:
        node = OpenCVCalibrationNode(
            boards, calibration_data_path, cam_shot_path, calib_flags, fisheye_calib_flags,
            pattern, options.camera_name, checkerboard_flags=checkerboard_flags,
            max_chessboard_speed=options.max_chessboard_speed, queue_size=options.queue_size)
        if mode == "mono":
            mono_thread = CaptureMono(options.sources, node.queue_monocular, height=options.height, width=options.width)
            mono_thread.start()
        elif mode == "stereo":
            stereo_thread = CaptureStereo(options.sources, node.queue_stereo, height=options.height, width=options.width)
            stereo_thread.start()
        else:
            logger.error(f"Unsupported camera model: {mode}. Please use 'mono' or 'stereo'.")
            return
        node.spin()
        # while True:
        #     time.sleep(0.01)
    finally:
        stop_event().set()



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        # print(e)