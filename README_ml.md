# Drone Trajectory Prediction - Machine Learning Model

This document details the Machine Learning model designed to predict the quadcopter drone's future states based on its current states and thruster inputs.

---

## Predictive Strategy (Recursive State Delta Prediction)

Instead of predicting the next absolute state vector $\mathbf{s}_{t+1}$ directly, the neural network is trained to predict the change in state (the derivative/delta) over a timestep $\Delta t$:

$$\Delta \mathbf{s}_t = \mathbf{s}_{t+1} - \mathbf{s}_t$$

### Why Predict Deltas?
1. **Mathematical Stability**: The physical changes in position and velocity between tiny simulation steps ($\Delta t = 0.005$ s) are very small. Predicting absolute coordinates makes it easy for a neural network to trivially learn the identity function ($\mathbf{s}_{t+1} \approx \mathbf{s}_t$) without learning any actual physics. Predicting the delta forces the network to learn the underlying differential equations (acceleration and velocity rates).
2. **Standardization**: By scaling both inputs and targets to zero mean and unit variance, we prevent numerical underflow and speed up training convergence.

---

## Model Architecture

The model is built using scikit-learn's `MLPRegressor` multi-output Neural Network:

- **Input Features (14 dimensions)**:
  - Current state $\mathbf{s}_t$ (10 dimensions: $x, v_x, y, v_y, z, v_z, \phi, \omega_{\phi}, \theta, \omega_{\theta}$)
  - Current propeller forces $\mathbf{a}_t$ (4 dimensions: $T_1, T_2, T_3, T_4$)
- **Network Structure**:
  - Hidden Layer 1: 128 neurons (ReLU activation)
  - Hidden Layer 2: 64 neurons (ReLU activation)
- **Output Layer (10 dimensions)**:
  - Predicted state changes $\Delta \mathbf{s}_t$
- **Training Parameters**:
  - Solver: Adam optimizer (Initial learning rate: 0.001)
  - Batch Size: 256
  - Early Stopping: Enabled if validation loss does not improve for 10 epochs.

---

## Dataset Generation

The dataset is generated dynamically by executing the physics engine in the background:
- The script runs multiple simulation runs with randomized parameters.
- **Randomizations**: Target coordinates, initial velocities, initial tilt angles, and controller PID gains are randomly perturbed to force the drone to execute a wide variety of maneuvers and recover from wild orientations.
- Features and targets are extracted step-by-step from the generated trajectories.

---

## Recursive (Autoregressive) Trajectory Forecasting

To test the model's accuracy, we perform **recursive forecasting**:
1. We feed the model the drone's true initial state $\mathbf{s}_0$ and the true thruster command $\mathbf{a}_0$.
2. The model predicts $\Delta \mathbf{s}_0$, and we compute the next predicted state: $\mathbf{\hat{s}}_1 = \mathbf{s}_0 + \Delta \mathbf{s}_0$.
3. For all subsequent steps, we use the model's *own previous prediction* $\mathbf{\hat{s}}_t$ along with the actual thruster input $\mathbf{a}_t$ to predict the next state:

$$\mathbf{\hat{s}}_{t+1} = \mathbf{\hat{s}}_t + \text{Model}(\mathbf{\hat{s}}_t, \mathbf{a}_t)$$

This autoregressive loop evaluates whether the ML model has truly generalized the laws of motion or if small prediction errors accumulate and cause the trajectory to drift.

---

## CLI Usage Guide

Run the ML script with various options:

```bash
# 1. Generate data, train model, and recursively test on the 'diagonal' path
python drone_ml_predictor.py --num-runs 60 --steps-per-run 1200

# 2. Run test headlessly and save the evaluation plot as an image
python drone_ml_predictor.py --load-model --test-scenario wild_spin --no-plot

# 3. Export raw data as numpy arrays (X_data.npy, y_data.npy) without training
python drone_ml_predictor.py --generate-only --num-runs 100
```
