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
        """Initialize a mosaic canvas for displaying multiple patches.
        
        Args:
            patch_size: Size of each patch in pixels (default: 256).
            num_patches: Total number of patches to display (default: 100).
            downscale: Factor to downscale patches by (default: 4).
            patches_per_row: Number of patches per row (default: 10).
            bg_color: Background color as RGB tuple (default: (0, 0, 0)).
            alpha: Alpha transparency value, -1 for no transparency (default: -1).
        """
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
        """Reset the current coordinate to the top-left corner."""
        self.coord = np.array([0, 0])

    def increment_coord(self):
        """Move to the next patch position, wrapping to next row if needed."""
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
        """Save the canvas to a file.
        
        Args:
            save_path: Path to save the canvas image.
            **kwargs: Additional arguments passed to Image.save().
        """
        self.canvas.save(save_path, **kwargs)

    def paste_patch(self, patch):
        """Paste a patch onto the canvas at the current position.
        
        Args:
            patch: PIL Image patch to paste.
        """
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
        """Get the canvas image.
        
        Returns:
            PIL Image of the canvas.
        """
        return self.canvas


class ContourCheckingFunction(object):
    def __call__(self, pt):
        """Check if a point is inside the contour.
        
        Args:
            pt: Point coordinates.
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError


class IsInContourV1(ContourCheckingFunction):
    def __init__(self, contour):
        """Initialize with a contour.
        
        Args:
            contour: OpenCV contour array.
        """
        self.contour = contour

    def __call__(self, pt):
        """Check if point is inside contour.
        
        Args:
            pt: Point coordinates as tuple or array.
            
        Returns:
            1 if point is inside contour, 0 otherwise.
        """
        return (
            1
            if cv2.pointPolygonTest(self.contour, tuple(np.array(pt).astype(float)), False)
            >= 0
            else 0
        )


class IsInContourV2(ContourCheckingFunction):
    def __init__(self, contour, patch_size):
        """Initialize with a contour and patch size.
        
        Args:
            contour: OpenCV contour array.
            patch_size: Size of patches in pixels.
        """
        self.contour = contour
        self.patch_size = patch_size

    def __call__(self, pt):
        """Check if patch center point is inside contour.
        
        Args:
            pt: Top-left corner coordinates of patch.
            
        Returns:
            1 if patch center is inside contour, 0 otherwise.
        """
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
        """Initialize with contour, patch size, and center shift.
        
        Args:
            contour: OpenCV contour array.
            patch_size: Size of patches in pixels.
            center_shift: Shift from center as fraction of half patch size (default: 0.5).
        """
        self.contour = contour
        self.patch_size = patch_size
        self.shift = int(patch_size // 2 * center_shift)

    def __call__(self, pt):
        """Check if at least one of four corner points is inside contour.
        
        Args:
            pt: Top-left corner coordinates of patch.
            
        Returns:
            1 if any corner point is inside contour, 0 otherwise.
        """
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
        """Initialize with contour, patch size, and center shift.
        
        Args:
            contour: OpenCV contour array.
            patch_size: Size of patches in pixels.
            center_shift: Shift from center as fraction of half patch size (default: 0.5).
        """
        self.contour = contour
        self.patch_size = patch_size
        self.shift = int(patch_size // 2 * center_shift)

    def __call__(self, pt):
        """Check if all four corner points are inside contour.
        
        Args:
            pt: Top-left corner coordinates of patch.
            
        Returns:
            1 if all corner points are inside contour, 0 otherwise.
        """
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
