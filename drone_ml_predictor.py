#!/usr/bin/env python3
"""
Drone Trajectory Predictor using Machine Learning.

This script references and imports the physics simulator in 'drone_simulation.py'.
It performs the following steps:
1. Data Generation: Runs multiple randomized simulation runs to collect states and thrusters.
2. Feature Engineering: Prepares features [state_t, thrusters_t] and targets [state_{t+1} - state_t].
3. Model Training: Trains a multi-output Neural Network (MLPRegressor) using scikit-learn.
4. Trajectory Forecasting: Forecasts the drone trajectory recursively (autoregressively) and 
   plots the comparison between actual physics simulation and the ML model prediction.
"""

import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import the simulator from drone_simulation.py
try:
    from drone_simulation import DroneSimulation
except ImportError:
    print("Error: Could not import DroneSimulation from drone_simulation.py.")
    print("Please make sure drone_simulation.py is in the same directory.")
    import sys
    sys.exit(1)


def generate_dataset(num_runs=50, steps_per_run=2000):
    """
    Generate training data by running the drone simulator under various conditions
    with randomized initial states and target positions.
    """
    print(f"Generating data from {num_runs} randomized simulation runs...")
    
    X_list = []
    y_list = []
    
    # Preset scenarios we can draw from or randomize
    scenarios = ["straight_up", "horizontal", "diagonal", "wild_spin", "toss_back", "throw_down"]
    
    for run in range(num_runs):
        # Pick a random scenario to initialize states
        scene_name = np.random.choice(scenarios)
        
        # Instantiate simulator
        sim = DroneSimulation(scenario=scene_name)
        
        # Randomize target and parameters slightly to create rich, diverse trajectories
        random_target = (
            np.random.uniform(-20, 20),
            np.random.uniform(-20, 20),
            np.random.uniform(1, 15)
        )
        sim.config['target_pos'] = random_target
        
        # Randomize gains slightly
        sim.config['gains_pos_x'] = (np.random.uniform(0.3, 0.8), np.random.uniform(0.8, 1.3))
        sim.config['gains_pos_y'] = (np.random.uniform(0.3, 0.8), np.random.uniform(0.8, 1.3))
        sim.config['gains_pos_z'] = (np.random.uniform(0.8, 1.5), np.random.uniform(1.2, 1.8))
        
        # Limit steps to prevent excessively long simulations
        sim.steps = steps_per_run
        
        # Run the simulator
        sim.run()
        
        # Extract states and thruster forces
        # states: shape (steps, 10)
        # thrusters: shape (steps, 4)
        states = sim.states
        thrusters = sim.thrusters
        
        # Build features: current state s_t and current thruster input a_t
        # Target: state delta s_{t+1} - s_t
        n_samples = len(states) - 1
        if n_samples <= 0:
            continue
            
        # Feature vector at t: s_t (10 cols) + a_t (4 cols) = 14 dimensions
        features = np.column_stack((states[:-1], thrusters[:-1]))
        
        # Target vector at t: s_{t+1} - s_t = 10 dimensions
        targets = states[1:] - states[:-1]
        
        X_list.append(features)
        y_list.append(targets)
        
    X = np.vstack(X_list)
    y = np.vstack(y_list)
    
    print(f"Dataset generated successfully! Total samples: {X.shape[0]}")
    print(f"X shape (features): {X.shape}, y shape (targets): {y.shape}")
    return X, y


def train_model(X, y, model_path="drone_ml_model.pkl"):
    """
    Train a multi-output MLP neural network model to predict state changes.
    """
    print("\n--- Training Machine Learning Model ---")
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features for neural network stability
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    
    # Standardize targets (state deltas are small, scaling helps convergence)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    
    # Define Neural Network (MLP Regressor)
    model = MLPRegressor(
        hidden_layer_sizes=(256, 256, 128),  # Bigger capacity
        activation='relu',
        solver='adam',
        max_iter=1000,                        # Allow more iterations
        batch_size=512,
        learning_rate_init=0.001,
        tol=1e-6,                             # Tighter tolerance to prevent early stopping
        n_iter_no_change=20,                  # Wait longer before giving up
        random_state=42,
        verbose=True
    )

    
    # Train the model
    model.fit(X_train_scaled, y_train_scaled)
    
    # Evaluate
    train_score = model.score(X_train_scaled, y_train_scaled)
    val_score = model.score(X_val_scaled, y_val_scaled)
    print(f"\nTraining R^2 Score: {train_score:.4f}")
    print(f"Validation R^2 Score: {val_score:.4f}")
    
    # Save the model and scalers
    saved_data = {
        'model': model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }
    with open(model_path, 'wb') as f:
        pickle.dump(saved_data, f)
        
    print(f"Model and scalers saved successfully to '{model_path}'")
    return saved_data


def evaluate_model(model_data, scenario="diagonal", show_plot=True):
    """
    Evaluates the model by running a test scenario, and forecasting the trajectory 
    recursively step-by-step using ONLY the initial state and thruster inputs.
    """
    print(f"\n--- Evaluating Model on Scenario '{scenario}' ---")
    
    # 1. Run actual physics simulation
    sim = DroneSimulation(scenario=scenario)
    sim.run()
    
    true_states = sim.states       # Shape: (N, 10)
    true_thrusters = sim.thrusters # Shape: (N, 4)
    dt = sim.dt
    time_steps = len(true_states)
    
    # Unpack model and scaling parameters
    model = model_data['model']
    scaler_X = model_data['scaler_X']
    scaler_y = model_data['scaler_y']
    
       # 2. Run Single-Step prediction (instead of recursive)
    predicted_states = np.zeros_like(true_states)
    predicted_states[0] = true_states[0]
    
    for t in range(time_steps - 1):
        # We use the TRUE state from physics, not our previous prediction!
        current_state_true = true_states[t] 
        actual_thruster = true_thrusters[t]
        
        feature = np.hstack((current_state_true, actual_thruster)).reshape(1, -1)
        feature_scaled = scaler_X.transform(feature)
        delta_scaled = model.predict(feature_scaled)
        delta = scaler_y.inverse_transform(delta_scaled.reshape(1, -1)).flatten()
        
        predicted_states[t + 1] = current_state_true + delta

        
    # Calculate Mean Squared Error (MSE) on coordinates
    pos_true = true_states[:, [0, 2, 4]] # X, Y, Z
    pos_pred = predicted_states[:, [0, 2, 4]]
    pos_mse = np.mean((pos_true - pos_pred) ** 2)
    print(f"Single-Step Trajectory Prediction Position MSE: {pos_mse:.6f} meters^2")
    
    # Calculate Mean Absolute Error (MAE)
    pos_mae = np.mean(np.abs(pos_true - pos_pred))
    print(f"Average Position Prediction Error (MAE): {pos_mae:.4f} meters")
    
    # 3. Plot Comparison Graph
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(f"ML Trajectory Prediction vs Physics Ground Truth\nScenario: '{scenario}' (Single-Step / Teacher Forcing)", fontsize=14)
    
    time_arr = np.arange(time_steps) * dt
    
    # XYZ Positions
    coord_labels = ['X Position', 'Y Position', 'Z Position']
    state_indices = [0, 2, 4]
    for idx, (label, state_idx) in enumerate(zip(coord_labels, state_indices)):
        ax = axes[idx, 0]
        ax.plot(time_arr, true_states[:, state_idx], 'g-', label='Physics (Ground Truth)', linewidth=2)
        ax.plot(time_arr, predicted_states[:, state_idx], 'r--', label='ML Prediction (Single-Step)', linewidth=1.5)
        ax.set_ylabel(f"{label} (m)")
        ax.grid(True)
        if idx == 0:
            ax.legend(loc='upper right')
        if idx == 2:
            ax.set_xlabel("Time (seconds)")
            
    # Pitch & Roll Angles
    angle_labels = ['Pitch (phi)', 'Roll (theta)']
    angle_indices = [6, 8]
    for idx, (label, state_idx) in enumerate(zip(angle_labels, angle_indices)):
        ax = axes[idx, 1]
        ax.plot(time_arr, np.degrees(true_states[:, state_idx]), 'g-', label='Physics', linewidth=2)
        ax.plot(time_arr, np.degrees(predicted_states[:, state_idx]), 'r--', label='ML Prediction', linewidth=1.5)
        ax.set_ylabel(f"{label} (deg)")
        ax.grid(True)
        if idx == 1:
            ax.set_xlabel("Time (seconds)")
            
    # Hide the unused 3rd subplot on the right column
    fig.delaxes(axes[2, 1])
    
    plt.tight_layout()
    plot_name = f"prediction_comparison_{scenario}.png"
    plt.savefig(plot_name, dpi=150)
    print(f"Comparison plot saved as '{plot_name}'")
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Drone Trajectory ML Predictor")
    parser.add_argument(
        "--generate-only", 
        action="store_true", 
        help="Only generate data, do not train"
    )
    parser.add_argument(
        "--num-runs", 
        type=int, 
        default=60, 
        help="Number of random simulation runs for training data (default: 60)"
    )
    parser.add_argument(
        "--steps-per-run", 
        type=int, 
        default=1200, 
        help="Timesteps per simulation run (default: 1200)"
    )
    parser.add_argument(
        "--model-path", 
        type=str, 
        default="drone_ml_model.pkl", 
        help="File path to save/load the trained model (default: drone_ml_model.pkl)"
    )
    parser.add_argument(
        "--test-scenario", 
        type=str, 
        default="diagonal", 
        choices=["straight_up", "horizontal", "diagonal", "wild_spin", "toss_back", "throw_down"],
        help="Preset scenario to test on after training (default: diagonal)"
    )
    parser.add_argument(
        "--load-model", 
        action="store_true", 
        help="Load pre-trained model and skip training"
    )
    parser.add_argument(
        "--no-plot", 
        action="store_true", 
        help="Disable showing interactive comparison plot (useful for headless verification)"
    )
    
    args = parser.parse_args()
    
    model_data = None
    
    if args.load_model:
        if os.path.exists(args.model_path):
            print(f"Loading pre-trained model from '{args.model_path}'...")
            with open(args.model_path, 'rb') as f:
                model_data = pickle.load(f)
        else:
            print(f"Error: Model file '{args.model_path}' not found. Cannot load.", file=sys.stderr)
            return
    else:
        # Step 1: Generate dataset
        X, y = generate_dataset(num_runs=args.num_runs, steps_per_run=args.steps_per_run)
        
        if args.generate_only:
            # Save raw dataset
            np.save("X_data.npy", X)
            np.save("y_data.npy", y)
            print("Dataset saved as X_data.npy and y_data.npy. Exiting.")
            return
            
        # Step 2: Train Model
        model_data = train_model(X, y, model_path=args.model_path)
        
    # Step 3: Evaluate Model
    if model_data is not None:
        evaluate_model(model_data, scenario=args.test_scenario, show_plot=not args.no_plot)


if __name__ == "__main__":
    main()
