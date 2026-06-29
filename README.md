# 3D Quadcopter Physics Simulator & 3D Trajectory Visualizer

A clean, modular Python-based physics simulation and interactive 3D visualization of a quadcopter flight controller. 

This project simulates a quadcopter drone transitioning from initial states (including high-velocity and wild spinning conditions) to target spatial positions, showcasing a cascade controller (outer position loop coupled with an inner attitude/rotation rate loop).

---

## Features

- **3D Physics Modeling**: Implements rigid-body quadcopter dynamics including linear and angular drag.
- **Cascade Control Architecture**:
  - **Outer Loop**: Clamped PD position controller yielding desired acceleration vectors.
  - **Coupling Stage**: Derives the required thrust magnitude and target pitch/roll angles to direct the thrust vector.
  - **Inner Loop**: PD attitude controller computing necessary angular accelerations.
  - **Actuator Mixer**: Distributes thrust commands to individual motors with clipping.
- **Vivid 3D Visualization**:
  - Animates the drone's spatial attitude (pitch and roll angles).
  - Displays propeller thrust differentials (deviation from balanced hover thrust) using dynamic, color-coded vectors.
  - Renders the total thrust vector dynamically.
  - Interactive camera control and trajectory path plotting.
- **Preset Scenarios**: Easy CLI testing for peaceful transitions, high-speed flights, and recovery from wild initial spins.

---

## Simulation Assumptions

To simplify the physics modeling, the following assumptions are made for the drone and environment:
- **Geometry & Mass Distribution**: The drone is modeled as a uniform flat disk, with a moment of inertia of $I = 0.5 \cdot M \cdot R^2$ (where $M$ is the mass and $R$ is the arm length).
- **Locked Axis of Rotation**: Yaw (rotation about the Z-axis) is locked/disabled. Only pitch ($\phi$) and roll ($\theta$) degrees of rotational freedom are simulated.
- **Actuator Response**: Propellers respond instantly to control inputs to produce torque (no motor response delay or transition dynamics).
- **Ideal Environment**: No external uncertainties, wind gusts, or sensor noise are simulated. The drone responds exactly as commanded and its state is known with absolute certainty.
- **Force Representation**: Propellers act as pure point force vectors rather than physical rotating objects (aerodynamic side-effects like blade element aerodynamics, ground effect, and gyroscopic precession are neglected).

---

## Preset Scenarios

- `straight_up`: Ascend straight up to `(0, 0, 10)` starting from rest at origin.
- `horizontal`: Travel horizontally to `(50, 0, 5)`.
- `diagonal`: Travel diagonally to `(10, -10, 10)`.
- `wild_spin`: Recover from a chaotic initial launch velocity and rapid spin, stabilizing safely at `(0, 0, 10)`.
- `toss_back`: Recover from forward/upward toss to stabilize at `(0, 0, -2)`.
- `throw_down`: Recover from downward and forward toss, stabilizing at `(0, -20, 1)`.

---

## Installation & Setup

### Prerequisites

Ensure you have a Python 3 environment. Install dependencies via pip:

```bash
pip install numpy matplotlib
```

*(Optional)* If you wish to save the animation to an MP4 video, make sure you have the `ffmpeg` executable installed on your system.

---

## Usage

Run the simulation with the default `diagonal` scenario:

```bash
python drone_simulation.py
```

### Command Line Arguments

Specify preset scenarios and display options:

```bash
# Run a specific scenario, e.g. the recovery from a wild spin
python drone_simulation.py --scenario wild_spin

# Run without showing the interactive GUI window (useful for automation/exports)
python drone_simulation.py --no-plot

# Save the animation to an interactive HTML file
python drone_simulation.py --save animation.html --no-plot

# Save the animation to an MP4 video (requires ffmpeg)
python drone_simulation.py --save animation.mp4 --ffmpeg-path "C:\Path\To\ffmpeg.exe"
```

---

## State Vector Architecture

The state vector is a 10-dimensional numpy array tracked at each timestep:

$$\mathbf{x} = \begin{bmatrix} x & v_x & y & v_y & z & v_z & \phi & \omega_{\phi} & \theta & \omega_{\theta} \end{bmatrix}^T$$

Where:
- $x, y, z$: Spatial positions in meters.
- $v_x, v_y, v_z$: Linear velocities in m/s.
- $\phi, \theta$: Pitch and roll orientation angles in radians.
- $\omega_{\phi}, \omega_{\theta}$: Pitch and roll angular rates in rad/s.

---

## Key Refactoring & Bug Fixes

This implementation corrects several bugs present in the original prototype notebook:

1. **Syntax Errors**: Removed syntax-breaking keywords (`YE`, `ye`) present on physics equations lines.
2. **Angle Visualization Bug**: Fixed visual orientation logic. Previously, the visualizer used pitch/roll rates ($\omega_\phi, \omega_\theta$) instead of actual angular positions ($\phi, \theta$) to compute frame rotation, causing a static drone rotation display.
3. **State Trimming Bug**: Replaced the filtering logic `np.any(states != 0, axis=1)` with explicit array slicing using the convergence index. The previous logic incorrectly deleted the initial state when the drone started at the origin with zero velocity.
4. **Modularity**: Converted script into a structured object-oriented class pattern suitable for importing or standalone CLI usage.

---

## Authors & Contributors

* **Alexander Ryu** - [@Aeryx-Git](https://github.com/Aeryx-Git)
* **Alexander Szumilo**

