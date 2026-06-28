# 3D Quadcopter Simulation & Machine Learning Trajectory Predictor

A modular Python framework that contains:
1. **Raw Math & Physics Simulator**: A quadcopter model governed by rigid-body flight dynamics, stabilized using a cascade PID controller, and visualized with Matplotlib 3D animations.
2. **Machine Learning Predictor**: A multi-output Neural Network trained on simulation telemetry to predict state updates ($\Delta s$) recursively.

---

## Detailed Documentation Guides

To read about the underlying mathematics, controllers, and models, select one of the following guides:

*   📖 **[Physics Simulator Guide (README_simulation.md)](README_simulation.md)**: Deep dive into the state vector, equations of motion, cascade controller (PID), mixer, and visualizer.
*   🧠 **[Machine Learning Guide (README_ml.md)](README_ml.md)**: Details on training dataset generation, Neural Network (MLP) architecture, feature scaling, and step-by-step recursive trajectory forecasting.

---

## Workspace Structure

- `drone_simulation.py`: Python module containing the physics engine, scenario presets, and Matplotlib 3D animation visualizer.
- `drone_ml_predictor.py`: Python script containing the ML pipeline (dataset generation, training loop, and recursive forecasting evaluation).
- `README_simulation.md`: Comprehensive documentation on raw physics and controller mathematics.
- `README_ml.md`: Comprehensive documentation on the MLP model design, data preparation, and training strategy.
- `.gitignore`: Configured to ignore python caches, checkpoints, and generated visual frames.

---

## Quick Start Setup

### 1. Install Dependencies
Make sure you have Python 3 installed. Install the required libraries:

```bash
pip install numpy matplotlib scikit-learn
```

### 2. Run the Physics Simulator
To run the raw PID trajectory simulator for the default diagonal flight path scenario:

```bash
python drone_simulation.py
```

### 3. Train and Evaluate the ML Model
To generate training data, fit the Neural Network, and plot the predicted trajectory against the physics model:

```bash
python drone_ml_predictor.py --num-runs 60 --steps-per-run 1200
```
*(This will save the trained model as `drone_ml_model.pkl` and save an evaluation chart comparing spatial positions and angles over time).*
