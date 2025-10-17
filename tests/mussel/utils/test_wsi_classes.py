import numpy as np
from PIL import Image

from mussel.utils.wsi_classes import Mosaic_Canvas


def test_mosaic_canvas_initialization():
    """Test Mosaic_Canvas initialization"""
    canvas = Mosaic_Canvas(patch_size=256, n=100, downscale=4, n_per_row=10)
    
    assert canvas.patch_size == 256
    assert canvas.downscaled_patch_size == 64
    assert canvas.n_rows == 10
    assert canvas.n_cols == 10
    assert canvas.dimensions[0] == 640  # 10 cols * 64 pixels
    assert canvas.dimensions[1] == 640  # 10 rows * 64 pixels


def test_mosaic_canvas_coord_increment():
    """Test coordinate increment in Mosaic_Canvas"""
    canvas = Mosaic_Canvas(patch_size=256, n=9, downscale=4, n_per_row=3)
    
    # Initial coordinates
    assert np.array_equal(canvas.coord, [0, 0])
    
    # First increment (move right)
    canvas.increment_coord()
    assert np.array_equal(canvas.coord, [canvas.downscaled_patch_size, 0])
    
    # Second increment (move right)
    canvas.increment_coord()
    assert np.array_equal(canvas.coord, [2 * canvas.downscaled_patch_size, 0])
    
    # Third increment (wrap to next row)
    canvas.increment_coord()
    assert np.array_equal(canvas.coord, [0, canvas.downscaled_patch_size])


def test_mosaic_canvas_paste_patch():
    """Test pasting patches to Mosaic_Canvas"""
    canvas = Mosaic_Canvas(patch_size=256, n=4, downscale=4, n_per_row=2)
    
    # Create a test patch
    test_patch = Image.new('RGB', (256, 256), color='red')
    
    initial_coord = canvas.coord.copy()
    canvas.paste_patch(test_patch)
    
    # Verify coordinate moved
    assert not np.array_equal(canvas.coord, initial_coord)


def test_mosaic_canvas_reset_coord():
    """Test resetting coordinates in Mosaic_Canvas"""
    canvas = Mosaic_Canvas(patch_size=256, n=4, downscale=4, n_per_row=2)
    
    # Move coordinates
    canvas.increment_coord()
    canvas.increment_coord()
    
    # Reset
    canvas.reset_coord()
    assert np.array_equal(canvas.coord, [0, 0])


def test_mosaic_canvas_with_alpha():
    """Test Mosaic_Canvas with alpha channel"""
    canvas = Mosaic_Canvas(
        patch_size=256, n=4, downscale=4, n_per_row=2, alpha=0.5
    )
    
    # Canvas should be RGBA mode with alpha
    assert canvas.canvas.mode == 'RGBA'


def test_mosaic_canvas_save(tmp_path):
    """Test saving Mosaic_Canvas to file"""
    canvas = Mosaic_Canvas(patch_size=256, n=4, downscale=4, n_per_row=2)
    
    output_file = tmp_path / "mosaic.png"
    canvas.save(output_file)
    
    assert output_file.exists()


def test_mosaic_canvas_get_painting():
    """Test getting the canvas image"""
    canvas = Mosaic_Canvas(patch_size=256, n=4, downscale=4, n_per_row=2)
    
    painting = canvas.get_painting()
    assert isinstance(painting, Image.Image)
    assert painting.size == tuple(canvas.dimensions)
