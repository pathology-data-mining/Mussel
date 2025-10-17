import cv2
import numpy as np
from PIL import Image


class MosaicCanvas(object):
    def __init__(
        self,
        patch_size=256,
        num_patches=100,
        downscale=4,
        patches_per_row=10,
        bg_color=(0, 0, 0),
        alpha=-1,
    ):
        self.patch_size = patch_size
        self.downscaled_patch_size = int(np.ceil(patch_size / downscale))
        self.n_rows = int(np.ceil(num_patches / patches_per_row))
        self.n_cols = patches_per_row
        canvas_width = self.n_cols * self.downscaled_patch_size
        canvas_height = self.n_rows * self.downscaled_patch_size
        if alpha < 0:
            canvas = Image.new(size=(canvas_width, canvas_height), mode="RGB", color=bg_color)
        else:
            canvas = Image.new(
                size=(canvas_width, canvas_height), mode="RGBA", color=bg_color + (int(255 * alpha),)
            )

        self.canvas = canvas
        self.dimensions = np.array([canvas_width, canvas_height])
        self.reset_coord()

    def reset_coord(self):
        self.coord = np.array([0, 0])

    def increment_coord(self):
        assert np.all(self.coord <= self.dimensions)
        if (
            self.coord[0] + self.downscaled_patch_size
            <= self.dimensions[0] - self.downscaled_patch_size
        ):
            self.coord[0] += self.downscaled_patch_size
        else:
            self.coord[0] = 0
            self.coord[1] += self.downscaled_patch_size

    def save(self, save_path, **kwargs):
        self.canvas.save(save_path, **kwargs)

    def paste_patch(self, patch):
        assert patch.size[0] == self.patch_size
        assert patch.size[1] == self.patch_size
        self.canvas.paste(
            patch.resize(
                tuple([self.downscaled_patch_size, self.downscaled_patch_size])
            ),
            tuple(self.coord),
        )
        self.increment_coord()

    def get_painting(self):
        return self.canvas


class ContourCheckingFunction(object):
    def __call__(self, pt):
        raise NotImplementedError


class IsInContourV1(ContourCheckingFunction):
    def __init__(self, contour):
        self.contour = contour

    def __call__(self, pt):
        return (
            1
            if cv2.pointPolygonTest(self.contour, tuple(np.array(pt).astype(float)), False)
            >= 0
            else 0
        )


class IsInContourV2(ContourCheckingFunction):
    def __init__(self, contour, patch_size):
        self.contour = contour
        self.patch_size = patch_size

    def __call__(self, pt):
        center_point = np.array(
            (pt[0] + self.patch_size // 2, pt[1] + self.patch_size // 2)
        ).astype(float)
        return (
            1
            if cv2.pointPolygonTest(self.contour, tuple(center_point), False)
            >= 0
            else 0
        )


# Easy version of 4pt contour checking function - 1 of 4 points need to be in the contour for test to pass
class IsInContourV3Easy(ContourCheckingFunction):
    def __init__(self, contour, patch_size, center_shift=0.5):
        self.contour = contour
        self.patch_size = patch_size
        self.shift = int(patch_size // 2 * center_shift)

    def __call__(self, pt):
        center = (pt[0] + self.patch_size // 2, pt[1] + self.patch_size // 2)
        if self.shift > 0:
            all_points = [
                (center[0] - self.shift, center[1] - self.shift),
                (center[0] + self.shift, center[1] + self.shift),
                (center[0] + self.shift, center[1] - self.shift),
                (center[0] - self.shift, center[1] + self.shift),
            ]
        else:
            all_points = [center]

        for point in all_points:
            if (
                cv2.pointPolygonTest(
                    self.contour, tuple(np.array(point).astype(float)), False
                )
                >= 0
            ):
                return 1
        return 0


# Hard version of 4pt contour checking function - all 4 points need to be in the contour for test to pass
class IsInContourV3Hard(ContourCheckingFunction):
    def __init__(self, contour, patch_size, center_shift=0.5):
        self.contour = contour
        self.patch_size = patch_size
        self.shift = int(patch_size // 2 * center_shift)

    def __call__(self, pt):
        center = (pt[0] + self.patch_size // 2, pt[1] + self.patch_size // 2)
        if self.shift > 0:
            all_points = [
                (center[0] - self.shift, center[1] - self.shift),
                (center[0] + self.shift, center[1] + self.shift),
                (center[0] + self.shift, center[1] - self.shift),
                (center[0] - self.shift, center[1] + self.shift),
            ]
        else:
            all_points = [center]

        for point in all_points:
            if (
                cv2.pointPolygonTest(
                    self.contour, tuple(np.array(point).astype(float)), False
                )
                < 0
            ):
                return 0
        return 1
