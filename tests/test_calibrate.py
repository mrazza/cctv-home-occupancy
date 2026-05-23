import os
import shutil
import tempfile
import pytest
from calibrate import calculate_arrow_endpoint, normalize_coordinates, update_config_file

def test_calculate_arrow_endpoint_horizontal():
    """Tests arrow endpoints calculation for simple horizontal segment."""
    # A = (100, 100), B = (200, 100) -> Vector v = (100, 0)
    # Left perpendicular: (0, 100). Mag: 100. Unit: (0, 1)
    # Midpoint C = (150, 100)
    # Arrow endpoint for length 50: C + 50 * (0, 1) = (150, 150)
    A = (100, 100)
    B = (200, 100)
    mid, arrow_dest = calculate_arrow_endpoint(A, B, length=50.0)
    
    assert mid == (150, 100)
    assert arrow_dest == (150, 150)

def test_calculate_arrow_endpoint_vertical():
    """Tests arrow endpoints calculation for simple vertical segment."""
    # A = (100, 200), B = (100, 100) -> Vector v = (0, -100)
    # Left perpendicular: (100, 0). Mag: 100. Unit: (1, 0)
    # Midpoint C = (100, 150)
    # Arrow endpoint for length 50: C + 50 * (1, 0) = (150, 150)
    A = (100, 200)
    B = (100, 100)
    mid, arrow_dest = calculate_arrow_endpoint(A, B, length=50.0)
    
    assert mid == (100, 150)
    assert arrow_dest == (150, 150)

def test_normalize_coordinates():
    """Tests pixel to normalized float conversion."""
    A = (100, 200)
    B = (300, 400)
    width = 1000
    height = 500
    
    res = normalize_coordinates(A, B, width, height)
    assert res == [(0.1, 0.4), (0.3, 0.8)]

def test_normalize_coordinates_invalid():
    with pytest.raises(ValueError):
        normalize_coordinates((10, 10), (20, 20), 0, 100)

def test_update_config_file():
    """Tests safe rewrite of the config file."""
    temp_dir = tempfile.mkdtemp()
    try:
        # Create a mock config file mirroring real structure
        mock_config_path = os.path.join(temp_dir, "config.py")
        original_content = """
class CameraConfig(BaseModel):
    rtsp_url: str = Field(default="rtsp://localhost")
    tripwire_line: list[tuple[float, float]] = Field(
        default=[(0.2, 0.5), (0.8, 0.5)],
        description="Coordinates of the tripwire"
    )
"""
        with open(mock_config_path, "w") as f:
            f.write(original_content)
            
        new_line = [(0.1234, 0.5678), (0.9876, 0.4321)]
        success = update_config_file(mock_config_path, new_line)
        
        assert success is True
        
        with open(mock_config_path, "r") as f:
            updated_content = f.read()
            
        # Verify the replacement worked cleanly
        assert "default=[(0.1234, 0.5678), (0.9876, 0.4321)]" in updated_content
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
