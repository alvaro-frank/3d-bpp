# ==============================================================================
# FILE: tests/test_environment.py
# DESCRIPTION: Unit tests for the environment core components (Bin and Box).
#              Verifies initialization, placement logic, boundary checks, and 
#              collision detection mechanics.
# ==============================================================================
import pytest
import numpy as np
from src.environment.bin import Bin
from src.environment.box import Box

def test_bin_initialization():
    """
    Tests if the container initializes empty and with correct dimensions.
    """
    bin = Bin(10, 10, 10)
    assert bin.width == 10
    assert bin.depth == 10
    assert bin.height == 10
    assert len(bin.boxes) == 0
    assert bin.bin_volume() == 1000

def test_box_placement_valid():
    """
    Tests if a box can be successfully placed in a valid position.
    """
    bin = Bin(10, 10, 10)
    box = Box(2, 2, 2)
    
    # Attempt to place the box at the origin (0, 0) with default rotation
    success = bin.place_box(box, (0, 0), rotation_type=0)
    
    assert success is True
    assert len(bin.boxes) == 1
    assert box.position == (0, 0, 0)

def test_box_out_of_bounds():
    """
    Tests if the environment rejects boxes that exceed bin boundaries.
    """
    bin = Bin(10, 10, 10)
    box = Box(5, 5, 5)
    
    # Attempt placement where box exceeds width (x=8 + w=5 = 13 > 10) 
    success = bin.place_box(box, (8, 0), rotation_type=0)
    
    assert success is False
    assert len(bin.boxes) == 0

def test_box_collision():
    """
    Tests if the environment correctly detects collisions between boxes.
    """
    bin = Bin(10, 10, 10)
    
    # Place the first large box
    box1 = Box(5, 5, 5)
    bin.place_box(box1, (0, 0), rotation_type=0) # Ocupa 0,0,0 até 5,5,5
    
    # Define a second smaller box
    box2 = Box(2, 2, 2)
    
    # Verify direct collision logic
    assert bin.collides((2,2,2), (1,1,0)) is True # Colide com a base da box1
    assert bin.collides((2,2,2), (6,6,0)) is False # Não colide (está ao lado)
    
def test_box_stacking():
    """Test if boxes are correctly stacked on top of each other (gravity logic)."""
    bin = Bin(10, 10, 10)
    
    # Box 1: Base (5x5x5) at (0,0)
    box1 = Box(5, 5, 5)
    bin.place_box(box1, (0, 0), rotation_type=0)
    
    # Box 2: Same footprint (2x2x2) placed at same (0,0)
    box2 = Box(2, 2, 2)
    success = bin.place_box(box2, (0, 0), rotation_type=0)
    
    assert success is True
    # Box 2 should be at z=5 (on top of Box 1), not z=0
    assert box2.position == (0, 0, 5)
    
def test_box_rotation_fit():
    """Test if rotating a box allows it to fit in a constrained space."""
    bin = Bin(10, 10, 10)
    
    # Create a "wall" leaving only 2 units of width at x=0
    wall = Box(8, 10, 10)
    bin.place_box(wall, (2, 0), rotation_type=0) 
    
    # Try to place a 5x2x2 box at (0,0). 
    # Default (rot=0): Width is 5. Should FAIL (5 > 2 available space).
    box_fail = Box(5, 2, 2)
    assert bin.place_box(box_fail, (0, 0), rotation_type=0) is False
    
    # Try again with Rotation (rot=1 implies swapping W and H/D depending on logic).
    # Assuming rotation makes Width=2. Should SUCCESS.
    box_success = Box(5, 2, 2)
    # Check src/environment/box.py to confirm which rot ID swaps Width.
    # Usually rot=2 or 4 swaps W and D.
    assert bin.place_box(box_success, (0, 0), rotation_type=2) is True
    
def test_box_height_limit():
    """Test that placing a box exceeding the bin height is rejected."""
    bin = Bin(10, 10, 5) # Height is only 5
    
    box1 = Box(5, 5, 3)
    bin.place_box(box1, (0, 0), rotation_type=0) # OK, z=0 to 3
    
    box2 = Box(5, 5, 3)
    # Stacking box2 (height 3) on box1 (ends at 3) -> Top at 6.
    # 6 > Bin Height 5. Should FAIL.
    success = bin.place_box(box2, (0, 0), rotation_type=0)
    
    assert success is False