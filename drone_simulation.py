#!/usr/bin/env python3
"""
3D Quadcopter Physics Simulation and 3D Trajectory Visualization.

This script simulates a 3D quadcopter drone controlled by:
1. An outer loop controller: computes desired acceleration and clamps it to target position.
2. A coupling stage: computes the total thrust magnitude and desired pitch/roll angles.
3. An inner loop controller: computes desired angular accelerations and translates them 
   into individual propeller thrusts.
4. Quadcopter equations of motion: updates the state vector using physics.

The simulation visualizes the drone trajectory, propeller thrust differentials, 
and total thrust vector using Matplotlib 3D animations.

Simulation Assumptions:
- **Geometry & Mass**: The drone is modeled as a uniform flat disk, with a moment of 
  inertia of I = 0.5 * M * R^2 (where M is mass and R is arm length).
- **Locked Rotation Axis**: Yaw (rotation about Z) is locked/disabled. Only pitch (phi) 
  and roll (theta) rotational degrees of freedom are simulated.
- **Actuator Response**: Propellers respond instantly to control inputs to produce torque.
- **Uncertainty**: No external wind, sensor noise, or environmental uncertainty.
- **Forces**: Propellers act as pure point force vectors rather than physical rotating 
  objects (no blade element dynamics or gyroscopic precession).
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

# Set matplotlib to use an interactive backend if run as script
# but allow fallback if no GUI is present
try:
    import matplotlib
    # Try using TkAgg for interactive window if available
    matplotlib.use('TkAgg')
except Exception:
    pass


class DroneSimulation:
    def __init__(self, scenario="diagonal", custom_config=None):
        """
        Initialize the drone simulation with a preset scenario or a custom configuration.
        
        Scenarios:
            - 'straight_up': Ascend straight up to (0, 0, 10) starting from rest.
            - 'horizontal': Fly horizontally to (50, 0, 5) starting from rest.
            - 'diagonal': Fly diagonally to (10, -10, 10) starting from rest.
            - 'wild_spin': Tossed in the air with high initial velocities and wild rotations, stabilizing at (0, 0, 10).
            - 'toss_back': Tossed forward/upward, returning to target (0, 0, -2).
            - 'throw_down': Thrown downward and forward, stabilizing at (0, -20, 1).
        """
        self.scenario = scenario
        self.dt = 0.005
        self.steps = 8000
        
        # Default drone physical and controller parameters
        self.config = {
            'mass': 2.0,                  # mass of the drone in kg
            'g': 9.81,                   # gravity acceleration in m/s^2
            'v_max': 5.0,                # max velocity (unused in raw ODE, but kept for reference)
            'thrust_minmax': (0.0, 1000.0), # thrust bounds per propeller (N)
            'n_prop': 4,                 # number of propellers
            'arm_length': 0.25,          # arm length from center to prop in meters
            'target_pos': (0.0, 0.0, 10.0), # target (x0, y0, z0)
            'gains_pos_x': (0.5, 1.0),   # (kp, kv) gains for X position control
            'gains_pos_y': (0.5, 1.0),   # (kp, kv) gains for Y position control
            'gains_pos_z': (1.0, 1.5),   # (kp, kv) gains for Z position control
            'k_drag_lin': 0.1,           # linear drag coefficient
            'k_drag_rot': 0.5,           # rotational drag coefficient
        }
        
        # Preset scenarios definitions
        self.scenarios = {
            'straight_up': {
                'camera_angle': (30, 165),
                'zoom_radius': 2.0,
                'speed': 1.0,
                'target': (0.0, 0.0, 10.0),
                'initial_pos': [0.0, 0.0, 0.0],
                'initial_vel': [0.0, 0.0, 0.0],
                'initial_pitch': [0.0, 0.0],
                'initial_roll': [0.0, 0.0]
            },
            'horizontal': {
                'camera_angle': (15, 90),
                'zoom_radius': 2.0,
                'speed': 1.0,
                'target': (50.0, 0.0, 5.0),
                'initial_pos': [0.0, 0.0, 0.0],
                'initial_vel': [0.0, 0.0, 0.0],
                'initial_pitch': [0.0, 0.0],
                'initial_roll': [0.0, 0.0]
            },
            'diagonal': {
                'camera_angle': (15, 140),
                'zoom_radius': 1.0,
                'speed': 0.7,
                'target': (10.0, -10.0, 10.0),
                'initial_pos': [0.0, 0.0, 0.0],
                'initial_vel': [0.0, 0.0, 0.0],
                'initial_pitch': [0.0, 0.0],
                'initial_roll': [0.0, 0.0]
            },
            'wild_spin': {
                'camera_angle': (15, -15),
                'zoom_radius': 2.0,
                'speed': 0.7,
                'target': (0.0, 0.0, 10.0),
                'initial_pos': [0.0, 0.0, 0.0],
                'initial_vel': [10.0, -10.0, 10.0],
                'initial_pitch': [20.0, 10.0],
                'initial_roll': [5.0, -20.0]
            },
            'toss_back': {
                'camera_angle': (10, 165),
                'zoom_radius': 3.0,
                'speed': 0.25,
                'target': (0.0, 0.0, -2.0),
                'initial_pos': [0.0, 0.0, 0.0],
                'initial_vel': [10.0, 0.0, 10.0],
                'initial_pitch': [0.0, 0.0],
                'initial_roll': [0.0, 2.0]
            },
            'throw_down': {
                'camera_angle': (10, 165),
                'zoom_radius': 4.0,
                'speed': 1.0,
                'target': (0.0, -20.0, 1.0),
                'initial_pos': [0.0, 0.0, 0.0],
                'initial_vel': [1.0, 12.0, -30.0],
                'initial_pitch': [0.0, 0.0],
                'initial_roll': [0.0, 0.0]
            }
        }
        
        # Load preset parameters or apply custom config override
        if scenario in self.scenarios:
            scene = self.scenarios[scenario]
            self.camera_angle = scene['camera_angle']
            self.zoom_radius = scene['zoom_radius']
            self.speed = scene['speed']
            self.config['target_pos'] = scene['target']
            
            # Pack initial state vector
            # States: [x, vx, y, vy, z, vz, phi, w_p, theta, w_r]
            # (phi = pitch, theta = roll)
            self.initial_state = np.array([
                scene['initial_pos'][0], scene['initial_vel'][0],
                scene['initial_pos'][1], scene['initial_vel'][1],
                scene['initial_pos'][2], scene['initial_vel'][2],
                *scene['initial_pitch'],
                *scene['initial_roll']
            ])
        else:
            raise ValueError(f"Unknown scenario '{scenario}'. Choose from {list(self.scenarios.keys())}")
            
        if custom_config:
            self.config.update(custom_config)

        # Simulation output variables
        self.t = np.linspace(0, self.steps * self.dt, self.steps + 1)
        self.states = np.zeros((self.steps + 1, 10))
        self.thrusters = np.zeros((self.steps + 1, 4))
        self.states[0] = self.initial_state
        self.end_snapshot = self.steps

    def ode_system_3d(self, state, params):
        """
        Computes derivatives and propeller force controls at the current time step.
        
        State vector layout:
          0: x      (position x)
          1: vx     (velocity x)
          2: y      (position y)
          3: vy     (velocity y)
          4: z      (position z)
          5: vz     (velocity z)
          6: phi    (pitch angle)
          7: w_p    (pitch rate / omega_phi)
          8: theta  (roll angle)
          9: w_r    (roll rate / omega_theta)
        """
        # Unpack state vector
        x, vx, y, vy, z, vz, phi, w_p, theta, w_r = state
        
        # Unpack physical & controller parameters
        mass, g = params['mass'], params['g']
        thrust_min, thrust_max = params['thrust_minmax']
        x0, y0, z0 = params['target_pos']
        num_propellers = params['n_prop']
        R = params['arm_length']
        kpx, kvx = params['gains_pos_x']
        kpy, kvy = params['gains_pos_y']
        kpz, kvz = params['gains_pos_z']
        
        # Moment of inertia (assuming simple thin rod model for arm structure)
        I = 0.5 * mass * R**2
        
        # 1. Outer Loop Controller: Compute Desired Accel in 3D (clamped for stability)
        ex, ey, ez = x - x0, y - y0, z - z0
        clip = 10.0
        ax_d = np.clip(-kpx * ex - kvx * vx, -clip, clip)
        ay_d = np.clip(-kpy * ey - kvy * vy, -clip, clip)
        az_d = np.clip(-kpz * ez - kvz * vz, -clip, clip)

        # 2. Coupling Stage: Compute Total Thrust and Target Orientation Angles
        # Required thrust to match desired accelerations, compensating for gravity
        F0 = mass * np.sqrt(ax_d**2 + ay_d**2 + (az_d + g)**2)
        
        # Target Pitch (phi_d) and Roll (theta_d) to direct the thrust vector
        phi_d = np.arctan2(-ax_d, az_d + g)
        theta_d = np.arctan2(-ay_d, az_d + g)

        # 3. Compute Rotational Error (using arctan2 to correctly wrap angular errors)
        err_phi = -np.arctan2(np.sin(phi - phi_d), np.cos(phi - phi_d))
        err_theta = -np.arctan2(np.sin(theta - theta_d), np.cos(theta - theta_d))
        
        # 4. Rotation Control Equations (PD controller with rotational drag compensation)
        S_P = 6.0  # Sensitivity to angular deviation
        S_D = 3.0  # Damping sensitivity to angular rate
        
        # Pitch Control
        dphi = w_p
        dw_p_calculated = (S_P) * err_phi - S_D * w_p - params['k_drag_rot'] * w_p * abs(w_p)
        # Required propeller differential force along X axis to generate desired pitch torque
        dF_x = dw_p_calculated * I / (2 * R)

        # Roll Control
        dtheta = w_r
        dw_r_calculated = (S_P) * err_theta - S_D * w_r - params['k_drag_rot'] * w_r * abs(w_r)
        # Required propeller differential force along Y axis to generate desired roll torque
        dF_y = dw_r_calculated * I / (2 * R)

        # 5. Propeller Thrust Allocation & Clamping
        # Allocating total thrust and differential forces to individual propellers
        # T_p1/T_p2 are on X-axis (pitch), T_t1/T_t2 are on Y-axis (roll)
        T_p1 = np.clip(F0 / num_propellers + dF_x, thrust_min, thrust_max)
        T_p2 = np.clip(F0 / num_propellers - dF_x, thrust_min, thrust_max)
        T_t1 = np.clip(F0 / num_propellers + dF_y, thrust_min, thrust_max)
        T_t2 = np.clip(F0 / num_propellers - dF_y, thrust_min, thrust_max)

        # Resulting angular accelerations from actual propeller forces
        dw_p = (T_p1 - T_p2) * R / I
        dw_r = (T_t1 - T_t2) * R / I

        # Total actual thrust generated
        sum_F = T_p1 + T_p2 + T_t1 + T_t2
        
        # Keep track of individual thrust values
        propeller_thrusts = np.array([T_p1, T_p2, T_t1, T_t2])

        # 6. Equations of Motion (State derivatives)
        dx = vx
        dvx = (-sum_F * np.sin(phi) * np.cos(theta) - params['k_drag_lin'] * vx) / mass
        dy = vy
        dvy = (-sum_F * np.sin(theta) - params['k_drag_lin'] * vy) / mass
        dz = vz
        dvz = (sum_F * np.cos(phi) * np.cos(theta) - mass * g - params['k_drag_lin'] * vz) / mass

        state_derivatives = np.array([dx, dvx, dy, dvy, dz, dvz, dphi, dw_p, dtheta, dw_r])
        return state_derivatives, propeller_thrusts

    def run(self):
        """Run the simulation loop until target is reached or step limit is reached."""
        t_x, t_y, t_z = self.config['target_pos']
        target_state = np.array([t_x, 0.0, t_y, 0.0, t_z, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        print(f"Starting simulation for scenario '{self.scenario}'...")
        print(f"Target position: ({t_x}, {t_y}, {t_z})")
        
        for i in range(self.steps):
            # Calculate derivatives and propeller forces
            state_update, thruster_update = self.ode_system_3d(self.states[i], self.config)
            
            # Update state with Euler method
            self.states[i + 1] = self.states[i] + state_update * self.dt
            self.thrusters[i + 1] = thruster_update
            
            # Check for convergence (all states close to target state)
            if np.isclose(self.states[i + 1], target_state, atol=1e-3).all():
                self.end_snapshot = i + 1
                print(f"Target reached and stabilized at step {i + 1} (t = {self.end_snapshot * self.dt:.3f} s).")
                break
                
        # Slice arrays to truncate unused pre-allocated elements
        self.states = self.states[:self.end_snapshot + 1]
        self.thrusters = self.thrusters[:self.end_snapshot + 1]
        self.t = self.t[:self.end_snapshot + 1]
        
        print(f"Simulation completed. Sliced data length: {len(self.states)} steps.")

    def animate(self, save_path=None, ffmpeg_path=None, show_plot=True):
        """
        Sets up the Matplotlib figure and runs the 3D visualizer animation.
        
        Arguments:
            save_path (str): File path to save output animation (.html or .mp4).
            ffmpeg_path (str): Optional override path for ffmpeg executable.
            show_plot (bool): If True, shows the interactive animation window.
        """
        # Graphing downsampling factor (graph every M-th simulation point)
        M = 10
        
        # Prepare snapshot variables for visual elements
        # NOTE: Fixed critical bug here by stacking orientation angles (indices 6 and 8)
        # instead of angular velocities (indices 7 and 9)
        angles_snapshot = np.column_stack((self.states[:, 6], self.states[:, 8]))[::M, :]
        xyz_snapshot = np.column_stack((self.states[:, 0], self.states[:, 2], self.states[:, 4]))[::M, :]
        thruster_snapshot = self.thrusters[::M, :]
        
        # Thrust differential calculations
        # Differences along X-axis (p1 - p2) and Y-axis (t1 - t2)
        thruster_diffs = np.column_stack((self.thrusters[:, 0] - self.thrusters[:, 1], 
                                          self.thrusters[:, 2] - self.thrusters[:, 3]))
        thruster_diff_snapshot = thruster_diffs[::M, :] / 2.0
        
        # Calculate how much each propeller deviates from balanced hover force
        thruster_diff_individual_prop = np.column_stack((
            -thruster_diff_snapshot[:, 0],  # Prop 1 deviation (negative X)
             thruster_diff_snapshot[:, 0],  # Prop 2 deviation (positive X)
            -thruster_diff_snapshot[:, 1],  # Prop 3 deviation (negative Y)
             thruster_diff_snapshot[:, 1]   # Prop 4 deviation (positive Y)
        ))
        
        # Total thrust vector snapshots pointing direction
        total_thrust_snapshot = thruster_snapshot.sum(axis=1, keepdims=True)
        thrust_x = total_thrust_snapshot * np.sin(angles_snapshot[:, 0, None]) * np.cos(angles_snapshot[:, 1, None])
        thrust_y = total_thrust_snapshot * -np.sin(angles_snapshot[:, 1, None])
        thrust_z = total_thrust_snapshot * np.cos(angles_snapshot[:, 0, None]) * np.cos(angles_snapshot[:, 1, None])

        # Drone geometry vectors representing the four motor arms
        vector_list = np.array([
            [1.0, 0.0, 0.0],   # Arm 1 (positive X)
            [-1.0, 0.0, 0.0],  # Arm 2 (negative X)
            [0.0, 1.0, 0.0],   # Arm 3 (positive Y)
            [0.0, -1.0, 0.0]   # Arm 4 (negative Y)
        ])
        
        # Calculate rotated arm vectors and thrust deviations
        arm_pos_x, arm_pos_y, arm_pos_z = [], [], []
        thrust_diff_x, thrust_diff_y, thrust_diff_z = [], [], []
        colors = []
        
        # Limits for scaling vector lengths on screen
        L_max, L_min = 1.0, 0.1
        
        # Vector visualization calculations for each arm
        for i in range(self.config['n_prop']):
            # Arm positions under current rotation angles (Euler yaw rotation is ignored in this simple model)
            phi_snap = angles_snapshot[:, 0]
            theta_snap = angles_snapshot[:, 1]
            
            x_arm = self.config['arm_length'] * (vector_list[i, 0] * np.cos(phi_snap) + 
                                                 vector_list[i, 1] * np.sin(phi_snap) * np.sin(theta_snap))
            y_arm = self.config['arm_length'] * (vector_list[i, 1] * np.cos(theta_snap))
            z_arm = self.config['arm_length'] * (vector_list[i, 0] * -np.sin(phi_snap) + 
                                                 vector_list[i, 1] * np.cos(phi_snap) * np.sin(theta_snap))
            
            arm_pos_x.append(x_arm)
            arm_pos_y.append(y_arm)
            arm_pos_z.append(z_arm)
            
            # Differential thrust vectors (clamped to prevent visual arrows from exploding in size)
            prop_dev = np.clip(thruster_diff_individual_prop[:, i], -20.0, 20.0)
            
            dx_diff = prop_dev * np.sin(phi_snap) * np.cos(theta_snap)
            dy_diff = prop_dev * -np.sin(theta_snap)
            dz_diff = prop_dev * np.cos(phi_snap) * np.cos(theta_snap)
            
            thrust_diff_x.append(dx_diff)
            thrust_diff_y.append(dy_diff)
            thrust_diff_z.append(dz_diff)
            
            # Normalize and apply color mapping (Jet color map based on absolute force deviation)
            norm_val = np.abs(prop_dev)
            min_val = max(norm_val.min(), 1e-5)
            max_val = max(norm_val.max(), 1e-1)
            colors.append(plt.cm.jet(mcolors.LogNorm(vmin=min_val, vmax=max_val)(norm_val)))
            
        # Scaling multiplier factor for thrust vector visualizations
        lengths = 4.0 * np.clip(3e-4 * np.abs(thruster_diff_individual_prop), L_min, L_max)

        # --- Figure and Static Setup ---
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        
        # Set titles
        ax.set_title(f"Quadcopter 3D Simulation: Scenario '{self.scenario}'", fontsize=12, pad=15)
        ax.view_init(elev=self.camera_angle[0], azim=self.camera_angle[1])
        ax.set_aspect('equal')

        # Draw full trajectory (static path)
        path_x, path_y, path_z = self.states[:, 0], self.states[:, 2], self.states[:, 4]
        ax.plot(path_x, path_y, path_z, 'b-', alpha=0.4, label='Drone Trajectory')
        
        # Draw Target marker
        t_x, t_y, t_z = self.config['target_pos']
        ax.scatter(t_x, t_y, t_z, color='red', s=100, marker='*', depthshade=False, label='Target Position')
        
        # Initialize active visual elements
        drone_dot, = ax.plot([], [], [], 'o', color='orange', markersize=10, zorder=10, label='Drone Center')
        
        # Initial empty quiver objects for drone frame arms (Green lines)
        drone_arms = [
            ax.quiver(0, 0, 0, 0, 0, 0, color='gray', linewidth=2, normalize=False, arrow_length_ratio=0) 
            for _ in range(4)
        ]
        
        # Initial empty quiver for total thrust force (Green thick arrow)
        total_thrust_vector = ax.quiver(0, 0, 0, 0, 0, 0, color='#11E73F', linewidth=3, normalize=True, arrow_length_ratio=0.15)
        
        # Initial empty quivers representing propeller differential thrusts (colored vectors)
        diff_thrust_vectors = [
            ax.quiver(0, 0, 0, 0, 0, 0, color='blue', normalize=False, arrow_length_ratio=0.1) 
            for _ in range(4)
        ]
        
        # Add legend
        ax.legend(loc='upper right')

        # Cache variables globally within this local scope for update function efficiency
        self._drone_dot = drone_dot
        self._drone_arms = drone_arms
        self._total_thrust_vector = total_thrust_vector
        self._diff_thrust_vectors = diff_thrust_vectors

        # Define Animation update frame
        def update_frame(frame):
            # Remove previous quiver drawing objects
            for arm in self._drone_arms:
                arm.remove()
            self._total_thrust_vector.remove()
            for force in self._diff_thrust_vectors:
                force.remove()
                
            # Get current coordinates
            x, y, z = xyz_snapshot[frame, 0], xyz_snapshot[frame, 1], xyz_snapshot[frame, 2]
            
            # Dynamically adjust camera limits to follow the drone (zoom box window)
            r = self.zoom_radius
            ax.set_xlim(x - r, x + r)
            ax.set_ylim(y - r, y + r)
            ax.set_zlim(z - r, z + r)
            ax.set_aspect('equal')
            
            # Update drone center dot
            self._drone_dot.set_data([x], [y])
            self._drone_dot.set_3d_properties([z])
            
            # Draw the 4 physical structure arms extending from center
            self._drone_arms = [
                ax.quiver(x, y, z, arm_pos_x[i][frame], arm_pos_y[i][frame], arm_pos_z[i][frame], 
                          length=1.0, color='black', linewidth=3, arrow_length_ratio=0)
                for i in range(4)
            ]
            
            # Draw total thrust vector centered at drone hub
            tx, ty, tz = thrust_x[frame], thrust_y[frame], thrust_z[frame]
            self._total_thrust_vector = ax.quiver(x, y, z, tx, ty, tz, 
                                                   length=0.4, color='#11E73F', linewidth=3, 
                                                   normalize=True, arrow_length_ratio=0.25)
            
            # Draw individual differential forces on each arm tip
            self._diff_thrust_vectors = [
                ax.quiver(x + arm_pos_x[i][frame], y + arm_pos_y[i][frame], z + arm_pos_z[i][frame], 
                          lengths[frame, i] * thrust_diff_x[i][frame], 
                          lengths[frame, i] * thrust_diff_y[i][frame], 
                          lengths[frame, i] * thrust_diff_z[i][frame], 
                          length=1.0, color=colors[i][frame], linewidth=2, arrow_length_ratio=0.2)
                for i in range(4)
            ]

        # Calculate appropriate visual speed interval
        # interval (ms) = 1000 * dt * M / play_speed
        anim_interval = 1000 * self.dt * M / self.speed
        num_frames = len(xyz_snapshot)
        
        anim = animation.FuncAnimation(
            fig, update_frame, frames=num_frames, 
            interval=anim_interval, blit=False, repeat=True
        )

        # Handling exports if save_path is requested
        if save_path:
            # Determine suffix type
            _, ext = os.path.splitext(save_path.lower())
            
            if ext == '.html':
                print(f"Saving animation to HTML file: {save_path}...")
                anim.save(save_path, writer='html')
                print("HTML file saved successfully.")
            elif ext == '.mp4':
                print(f"Saving animation to MP4 file: {save_path}...")
                # Configure ffmpeg path if override provided
                if ffmpeg_path:
                    plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path
                
                try:
                    writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='Drone Sim'), bitrate=1800)
                    anim.save(save_path, writer=writer)
                    print("MP4 file saved successfully.")
                except Exception as e:
                    print(f"Error saving to MP4: {e}", file=sys.stderr)
                    print("Falling back to HTML format...", file=sys.stderr)
                    fallback_html = save_path.replace('.mp4', '.html')
                    anim.save(fallback_html, writer='html')
                    print(f"HTML fallback saved to: {fallback_html}")
            else:
                print(f"Unsupported save extension '{ext}'. Only .html and .mp4 are supported.", file=sys.stderr)

        if show_plot:
            print("Rendering interactive window. Close window to end execution.")
            plt.show()
        else:
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="3D Quadcopter Physics Simulator & Visualizer")
    parser.add_argument(
        "--scenario", 
        type=str, 
        default="diagonal", 
        choices=["straight_up", "horizontal", "diagonal", "wild_spin", "toss_back", "throw_down"],
        help="Select simulation scenario preset (default: diagonal)"
    )
    parser.add_argument(
        "--save", 
        type=str, 
        default=None, 
        help="Save animation to file (supported: path/to/output.html or path/to/output.mp4)"
    )
    parser.add_argument(
        "--ffmpeg-path", 
        type=str, 
        default=None, 
        help="Optional absolute path to ffmpeg executable (needed for MP4 save)"
    )
    parser.add_argument(
        "--no-plot", 
        action="store_true", 
        help="Disable interactive display (useful for script/batch exports)"
    )
    
    args = parser.parse_args()
    
    # Run simulation
    sim = DroneSimulation(scenario=args.scenario)
    sim.run()
    
    # Run visualization
    sim.animate(
        save_path=args.save, 
        ffmpeg_path=args.ffmpeg_path, 
        show_plot=not args.no_plot
    )


if __name__ == "__main__":
    main()
