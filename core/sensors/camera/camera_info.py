from enum import StrEnum
from typing import Literal

import cv2.typing
import numpy as np


class DistortionModels(StrEnum):
    PLUMB_BOB = "plumb_bob"
    RATIONAL_POLYNOMIAL = "rational_polynomial"


class CameraInfo:
    def __init__(self,
                 height: int=0,
                 width: int=0,
                 distortion_model: Literal["plumb_bob", "rational_polynomial"] = 'plumb_bob',
                 d = None,
                 k = None,
                 r = None,
                 p = None,
                 roi = None,
                 binding_x = 0,
                 binding_y = 0):
        """
        A class to store camera intrinsic and distortion parameters.

        Parameters:
            height (int): The image height in pixels.
            width (int): The image width in pixels.
            distortion_model (str): The distortion model ('plumb_bob' or 'rational_polynomial').
            d (np.array): Distortion coefficients.
            k (np.array): Intrinsic matrix (3x3).
            r (np.array): Rectification matrix (3x3).
            p (np.array): Projection matrix (3x4).
        """
        # Calibration Parameters

        # Internal parameters
        # The image dimensions with which the camera was calibrated.
        # Normally this will be the full camera resolution in pixels
        self.height = height
        self.width = width

        # Distortion model
        if distortion_model not in {"plumb_bob", "rational_polynomial"}:
            raise ValueError("Invalid distortion model. Use 'plumb_bob' or 'rational_polynomial'.")
        self.distortion_model = distortion_model

        # Calibration parameters
        self.d = d  # Distortion coefficients
        self.k = k  # Intrinsic matrix
        self.r = r  # Rectification matrix
        self.p = p  # Projection matrix

        # Binding offsets (default to tiny values)
        self.binding_x = binding_x
        self.binding_y = binding_y

        # The default setting of roi (all values 0) is considered the same as
        # full resolution (roi.width = width, roi.height = height).
        self.roi = cv2.typing.Rect(0, 0, width, height) if roi is None else roi

    def __repr__(self):
        return (f"CameraInfo(height={self.height}, width={self.width}, "
                f"distortion_model='{self.distortion_model}', d={self.d.tolist()}, "
                f"k={self.k.tolist()}, r={self.r.tolist()}, p={self.p.tolist()}, "
                f"binding_x={self.binding_x}, binding_y={self.binding_y}, roi={self.roi})")



if __name__ == '__main__':
    c = CameraInfo()
    print(c.distortion_model)
