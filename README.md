# RayTracer

A Python-based ray tracing simulation library for modeling optical systems and geometric ray paths through 3D space.

## Overview

This project provides a framework for simulating ray tracing through optical systems. It supports various geometric objects including planes, spheres, and paraboloids, with comprehensive coordinate transformation capabilities. The library can be used to model complex optical systems such as folded mirrors, spherical mirrors, and detector arrays.

### Key Features

- **Ray Tracing Engine**: Trace rays through 3D optical systems with accurate geometric computations
- **Geometric Objects**: Support for Plane, Sphere, and Paraboloid surfaces
- **Coordinate Transformations**: Full affine transformation support for positioning and orienting optical elements
- **Rotations**: Built-in rotation utilities for common transformations
- **Visualization**: 3D plotting of ray traces and optical systems
- **Type Safety**: Full mypy type checking with strict mode

## Installation

### Prerequisites

- Python 3.11 or higher
- pip or uv package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd RayTracer
```

2. Install dependencies:

Using pip:
```bash
pip install -r requirements.txt
```

Using uv:
```bash
uv sync
```

3. Install development dependencies (optional):
```bash
uv sync --group dev
```

## Quick Start

### Running the Example

The project includes a complete example demonstrating a point source being folded off a flat mirror and focused by a spherical mirror:

```bash
python src/main.py
```

This example:
- Creates a 200-ray cone from a point source
- Traces rays through a fold mirror (45° tilt)
- Reflects rays off a spherical mirror (24.0 unit radius)
- Focuses the beam onto a detector plane
- Reports the RMS spot radius and displays a 3D visualization

### Basic Usage

```python
from Raytracer import Rays, RayTracerScene, perform_ray_tracing, plot_traced_rays
from RayTracerObject import Plane, Sphere
from CoordinateTransformations import AffineTransformation
import numpy as np

# Create rays
rays = Rays(
    origin=np.array([[0, 0, 0]]),
    direction=np.array([[0, 0, 1]])
)

# Create optical objects
plane = Plane(
    coordinate_system=AffineTransformation("Plane", "World"),
    name="detector",
    xExtent=1.0,
    yExtent=1.0
)

# Create scene
scene = RayTracerScene(objects=[plane])

# Perform ray tracing
_, intersection_history = perform_ray_tracing(rays, scene)

# Visualize results
plot_traced_rays(intersection_history, scene)
```

## Project Structure

```
RayTracer/
├── src/
│   ├── main.py                        # Example: fold and spherical mirror system
│   ├── Raytracer.py                   # Core ray tracing engine
│   ├── RayTracerObject.py             # Geometric object definitions
│   ├── CoordinateTransformations.py   # Affine transformation utilities
│   └── Geometry/
│       ├── __init__.py
│       └── Rotations.py               # Rotation matrices
├── tests/
│   ├── test_raytracer.py              # Ray tracing tests
│   ├── test_coordinate_transformations.py  # Coordinate system tests
│   └── test_rotations.py              # Rotation utility tests
├── pyproject.toml                     # Project configuration
├── requirements.txt                   # Core dependencies
├── uv.lock                           # Lock file for dependencies
└── README.md                          # This file
```

## Components

### Core Modules

#### `Raytracer.py`
The main ray tracing engine with:
- `Rays` class: Represents a collection of rays with origins and directions
- `RayTracerScene` class: Container for optical objects and coordinate systems
- `perform_ray_tracing()`: Traces rays through a scene
- `plot_traced_rays()`: 3D visualization of ray paths

#### `RayTracerObject.py`
Abstract base class `RayTracerObject` and implementations:
- `Plane`: Flat rectangular surfaces with configurable extent
- `Sphere`: Spherical mirror surfaces with radius parameter
- `Paraboloid`: Parabolic surfaces with shape parameter

#### `CoordinateTransformations.py`
Manages coordinate transformations:
- `AffineTransformation`: Maps points between coordinate systems
- Supports rotation, translation, and scaling
- Enables complex optical system assembly

#### `Geometry/Rotations.py`
Rotation utilities:
- `rotx(theta)`: Rotation matrix about x-axis
- `roty(theta)`: Rotation matrix about y-axis
- `rotz(theta)`: Rotation matrix about z-axis

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

The project uses several tools to maintain code quality:

#### Formatting and Linting

Format code with Ruff:
```bash
ruff format src/ tests/
```

Check and fix code style:
```bash
ruff check src/ tests/ --fix
```

#### Type Checking

Run mypy type checker in strict mode:
```bash
mypy src/
```

#### Pre-commit Hooks

Install pre-commit hooks:
```bash
pre-commit install
```

Run checks manually:
```bash
pre-commit run --all-files
```

### Configuration

- **Python version**: 3.11+
- **Line length**: 100 characters
- **Type checking**: Strict mode enabled
- **Dependencies**: Managed with uv/pip

## Dependencies

### Core Dependencies
- **numpy**: Numerical array operations and linear algebra
- **pillow**: Image processing
- **matplotlib**: 3D visualization and plotting
- **tqdm**: Progress bars
- **requests**: HTTP library
- **pydantic**: Data validation (≥2.13.4)
- **networkx**: Graph algorithms (≥3.6.1)

### Development Dependencies
- **pytest**: Testing framework
- **mypy**: Static type checker (≥1.18.2)
- **ruff**: Fast Python linter and formatter (≥0.16.2)
- **pre-commit**: Git hook management (≥4.6.1)

## Examples

### Simple Plane Ray Intersection

```python
import numpy as np
from RayTracerObject import Plane
from CoordinateTransformations import AffineTransformation

# Create a plane at the origin
transform = AffineTransformation("MyPlane", "World")
transform.set_translation(np.array([0, 0, 10]))
plane = Plane(transform, "detector", xExtent=5.0, yExtent=5.0)

# Check surface normal at a point
point = np.array([0, 0, 0])
normal = plane.surface_normal(point)
print(f"Surface normal: {normal}")
```

### Coordinate Transformation

```python
from CoordinateTransformations import AffineTransformation
from Geometry.Rotations import rotx
import numpy as np

# Create a transformation with rotation and translation
transform = AffineTransformation("Object", "World")
transform.set_rotation(rotx(np.deg2rad(45)))  # 45° tilt about x-axis
transform.set_translation(np.array([0, -12, 0]))  # Position at (0, -12, 0)

# Transform a point from local to world coordinates
local_point = np.array([0, 0, 0])
world_point = transform.get_local_to_world()(local_point)
```

## Performance Notes

- Ray tracing is vectorized using NumPy for efficient batch processing
- Supports tracing hundreds of rays simultaneously
- Coordinate transformations use matrix operations for speed
- Consider the number of rays and optical elements when planning large simulations

## Contributing

When contributing to this project:
1. Ensure all code passes type checking: `mypy src/`
2. Follow the style guidelines enforced by Ruff
3. Write tests for new functionality
4. Update documentation as needed

## License

[Add license information here]

## References

This project implements classical ray optics principles for optical system simulation and analysis.
