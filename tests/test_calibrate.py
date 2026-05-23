import os
import json
import pytest
from calibrate import calculate_arrow_endpoint, normalize_coordinates, update_config_file

def test_calculate_arrow_endpoint():
    A = (100, 100)
    B = (200, 100)
    mid, arrow_dest = calculate_arrow_endpoint(A, B, length=50)
    
    # Midpoint should be (150, 100)
    assert mid == (150, 100)
    # Directed vector is horizontal right.
    # Perpendicular pointing left (in screen coords, Y goes down, so left perpendicular points UP towards Y=0)
    # Upwards means subtracting from Y
    assert arrow_dest[0] == 150
    assert arrow_dest[1] == 150

def test_normalize_coordinates():
    A = (100, 200)
    B = (300, 400)
    width, height = 1000, 1000
    norm = normalize_coordinates(A, B, width, height)
    
    assert norm[0] == (0.1, 0.2)
    assert norm[1] == (0.3, 0.4)

def test_normalize_coordinates_invalid():
    with pytest.raises(ValueError):
        normalize_coordinates((0, 0), (1, 1), 0, 100)

def test_update_config_file(temp_dir):
    json_path = os.path.join(temp_dir, "test_config.json")
    line = [(0.123, 0.456), (0.789, 0.987)]
    
    # Test creation
    assert update_config_file(json_path, line) is True
    assert os.path.exists(json_path)
    
    with open(json_path, "r") as f:
        data = json.load(f)
        assert data["tripwire_line"] == [[0.123, 0.456], [0.789, 0.987]]

    # Test update preserves other keys
    with open(json_path, "w") as f:
        json.dump({"other_key": "other_value"}, f)
        
    assert update_config_file(json_path, line) is True
    with open(json_path, "r") as f:
        data = json.load(f)
        assert data["other_key"] == "other_value"
        assert data["tripwire_line"] == [[0.123, 0.456], [0.789, 0.987]]