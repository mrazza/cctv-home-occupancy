# Tripwire Calibration Tool Plan

This document outlines the design and implementation of the calibration utility, enabling intuitive visual setup of Nest Camera doorways.

## Core Features
1. **Interactive Line Drawing**: Use mouse clicks on a live frame to set the start (A) and end (B) of a tripwire.
2. **Left-Hand Visual Cue**: An arrow is drawn automatically pointing to the "Left-Hand Side" of the directed line (Inside/ENTER) to prevent wrong orientation setups.
3. **Normalized Coordinates**: Converts pixel clicks to resolution-independent float coordinates `[(x1, y1), (x2, y2)]` from `0.0` to `1.0`.
4. **Config Auto-Saver**: Replaces the `tripwire_line` parameter inside `src/config.py` automatically when pressing **[S]**.

## Mathematical Calculations (Unit-Tested)
- **Vector Normalization**:
  $$\vec{v} = B - A$$
- **Left Perpendicular Vector**:
  $$\vec{n} = (-v_y, v_x)$$
- **Arrow Destination**:
  $$D = C + \text{length} \cdot \frac{\vec{n}}{\|\vec{n}\|}$$
  where $C$ is the line's midpoint: $C = \frac{A + B}{2}$.

## Code Structure
- `calibrate.py`: Executable utility with segregated GUI and testable business logic functions.
- `tests/test_calibrate.py`: Headless tests verifying normalized calculations, perpendicular offsets, and safe config-rewriting.
