import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon

from mussel.utils.segment import (
    contours_to_polygon,
    get_native_size,
    grid_bounds,
    is_black_patch,
    is_white_patch,
    partition,
    scale_contour_dim,
    scale_geometry,
    scale_holes_dim,
)


def test_is_white_patch():
    """Test is_white_patch function"""
    # Create a white patch (low saturation)
    white_patch = np.ones((256, 256, 3), dtype=np.uint8) * 255
    assert is_white_patch(white_patch, satThresh=5) == True
    
    # Create a colored patch (high saturation)
    colored_patch = np.zeros((256, 256, 3), dtype=np.uint8)
    colored_patch[:, :, 0] = 255  # Red channel
    assert is_white_patch(colored_patch, satThresh=5) == False


def test_is_black_patch():
    """Test is_black_patch function"""
    # Create a black patch
    black_patch = np.zeros((256, 256, 3), dtype=np.uint8)
    assert is_black_patch(black_patch, rgbThresh=40) == True
    
    # Create a bright patch
    bright_patch = np.ones((256, 256, 3), dtype=np.uint8) * 200
    assert is_black_patch(bright_patch, rgbThresh=40) == False


def test_scale_geometry():
    """Test scale_geometry function"""
    # Create a simple polygon
    polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    
    # Scale by 2
    scaled = scale_geometry(polygon, 2.0)
    
    # Check that coordinates are scaled
    coords = list(scaled.exterior.coords)
    expected = [(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)]
    
    for actual, exp in zip(coords, expected):
        assert np.isclose(actual[0], exp[0])
        assert np.isclose(actual[1], exp[1])


def test_scale_contour_dim():
    """Test scale_contour_dim function"""
    contours = [
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[5.0, 6.0], [7.0, 8.0]])
    ]
    
    scaled = scale_contour_dim(contours, scale=(2.0, 3.0))
    
    assert len(scaled) == 2
    np.testing.assert_array_equal(scaled[0], np.array([[2, 6], [6, 12]], dtype='int32'))
    np.testing.assert_array_equal(scaled[1], np.array([[10, 18], [14, 24]], dtype='int32'))


def test_scale_holes_dim():
    """Test scale_holes_dim function"""
    holes = [
        [np.array([[1.0, 1.0]]), np.array([[2.0, 2.0]])],
        [np.array([[3.0, 3.0]])]
    ]
    
    scaled = scale_holes_dim(holes, scale=(2.0, 2.0))
    
    assert len(scaled) == 2
    assert len(scaled[0]) == 2
    np.testing.assert_array_equal(scaled[0][0], np.array([[2, 2]], dtype='int32'))


def test_get_native_size():
    """Test get_native_size function"""
    # Test with mpp equal to slide_mpp
    size = get_native_size(256, mpp=0.5, slide_mpp=0.5)
    assert size == 256
    
    # Test with mpp double the slide_mpp (should scale up)
    size = get_native_size(256, mpp=1.0, slide_mpp=0.5)
    assert size == 512
    
    # Test with mpp half the slide_mpp (should scale down)
    size = get_native_size(256, mpp=0.25, slide_mpp=0.5)
    assert size == 128


def test_grid_bounds():
    """Test grid_bounds function"""
    # Create a simple geometry
    polygon = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    
    # Create grid with step_size=50 and patch_size=50
    grid = grid_bounds(polygon, step_size=50, patch_size=50)
    
    # Should create a 2x2 grid (4 patches)
    assert len(grid) == 4
    
    # Check that all grid items are Polygons
    for item in grid:
        assert isinstance(item, Polygon)


def test_partition():
    """Test partition function"""
    # Create a polygon
    polygon = Polygon([(10, 10), (90, 10), (90, 90), (10, 90)])
    
    # Partition with step_size and patch_size
    grid = partition(polygon, step_size=30, patch_size=30)
    
    # All grid items should intersect with the polygon
    assert len(grid) > 0
    for item in grid:
        assert polygon.intersects(item)


def test_contours_to_polygon_simple():
    """Test contours_to_polygon with simple contours"""
    # Create simple square contour
    contour = np.array([[[0, 0]], [[100, 0]], [[100, 100]], [[0, 100]]])
    foreground_contours = [contour]
    
    polygon = contours_to_polygon(foreground_contours)
    
    assert isinstance(polygon, (Polygon, MultiPolygon))
    assert polygon.is_valid


def test_contours_to_polygon_with_holes():
    """Test contours_to_polygon with holes"""
    # Create outer contour
    outer = np.array([[[0, 0]], [[100, 0]], [[100, 100]], [[0, 100]]])
    
    # Create hole contour
    hole = np.array([[[25, 25]], [[75, 25]], [[75, 75]], [[25, 75]]])
    
    foreground_contours = [outer]
    hole_contours = [[hole]]
    
    polygon = contours_to_polygon(foreground_contours, hole_contours)
    
    assert isinstance(polygon, (Polygon, MultiPolygon))
    assert polygon.is_valid
