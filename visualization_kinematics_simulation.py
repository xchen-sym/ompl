import re
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description='Visualize trajectory data from log file')
parser.add_argument('log_file', help='Path to the log file containing trajectory data')
parser.add_argument('-o', '--output', 
                    default='figures/reference_vs_real_trajectory.png',
                    help='Output file path for the generated plot')
args = parser.parse_args()

# --- DATA CONTAINERS ---
path = []
traj = []
initial_waypoint = None
final_waypoint = None

# --- READ AND PARSE ---
with open(args.log_file, 'r') as f:
    in_path = False
    in_traj = False
    for line in f:
        line = line.strip()
        
        # Parse initial and final waypoints
        if line.startswith('Initial Waypoint:'):
            m = re.search(r'\[(.*)\]', line)
            if m:
                nums = [float(x) for x in m.group(1).split(',')]
                initial_waypoint = {'x': nums[0], 'y': nums[1], 'yaw': nums[2], 'v': nums[3], 'omega': nums[4]}
            continue
        if line.startswith('Final Waypoint:'):
            m = re.search(r'\[(.*)\]', line)
            if m:
                nums = [float(x) for x in m.group(1).split(',')]
                final_waypoint = {'x': nums[0], 'y': nums[1], 'yaw': nums[2], 'v': nums[3], 'omega': nums[4]}
            continue
            
        if line == 'Path:':
            in_path = True
            in_traj = False
            continue
        if line == 'Trajectory:':
            in_path = False
            in_traj = True
            continue

        # parse a bracketed list of numbers
        m = re.match(r'\[(.*)\]', line)
        if not m:
            # end of current section once non-bracket line encountered
            in_path = in_traj = False
            continue

        nums = [float(x) for x in m.group(1).split(',')]
        if in_path:
            # first two entries are x, y
            path.append((nums[0], nums[1]))
        elif in_traj:
            # entries are x, y, yaw, v, omega
            traj.append({'x': nums[0], 'y': nums[1], 'yaw': nums[2], 'v': nums[3], 'omega': nums[4]})

# --- EXTRACT COORDINATES ---
path_x, path_y = zip(*path)
traj_x = [p['x'] for p in traj]
traj_y = [p['y'] for p in traj]
traj_yaw = [p['yaw'] for p in traj]
traj_v = [p['v'] for p in traj]
traj_omega = [p['omega'] for p in traj]

# Time array (dt = 0.01s)
dt = 0.01
t = np.arange(len(traj)) * dt
t_final = t[-1] if len(t) > 0 else 0

# --- PLOT 1: X-Y TRAJECTORY ---
plt.figure(figsize=(8, 6))
plt.plot(path_x, path_y, 'o-', label=f'Reference Path: {0.01 * (len(path) - 1):.2f}s', markersize=3)
plt.plot(traj_x, traj_y, 'x-', label=f'Real Trajectory: {t_final:.2f}s', markersize=3)

# Add initial and final waypoints
if initial_waypoint:
    plt.plot(initial_waypoint['x'], initial_waypoint['y'], 'go', markersize=8, label='Initial Waypoint')
if final_waypoint:
    plt.plot(final_waypoint['x'], final_waypoint['y'], 'ro', markersize=8, label='Final Waypoint')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Reference Path vs. Real Trajectory')
plt.axis('equal')
plt.grid(True)
plt.legend()

# Save x-y plot
output_file = args.output
output_dir = os.path.dirname(output_file)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"X-Y plot saved to: {output_file}")
plt.close()

# --- PLOT 2: TEMPORAL PROFILES (X, Y, YAW) ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

# X vs time
ax1.plot(t, traj_x, 'b-', linewidth=2, label='Real Trajectory')
if initial_waypoint:
    ax1.plot(0, initial_waypoint['x'], 'go', markersize=8, label='Initial')
if final_waypoint:
    ax1.plot(t_final, final_waypoint['x'], 'ro', markersize=8, label='Target')
ax1.set_ylabel('x [m]')
ax1.grid(True)
ax1.legend()
ax1.set_title('Temporal Profiles: Position and Orientation')

# Y vs time
ax2.plot(t, traj_y, 'b-', linewidth=2, label='Real Trajectory')
if initial_waypoint:
    ax2.plot(0, initial_waypoint['y'], 'go', markersize=8, label='Initial')
if final_waypoint:
    ax2.plot(t_final, final_waypoint['y'], 'ro', markersize=8, label='Target')
ax2.set_ylabel('y [m]')
ax2.grid(True)
ax2.legend()

# Yaw vs time
ax3.plot(t, traj_yaw, 'b-', linewidth=2, label='Real Trajectory')
if initial_waypoint:
    ax3.plot(0, initial_waypoint['yaw'], 'go', markersize=8, label='Initial')
if final_waypoint:
    ax3.plot(t_final, final_waypoint['yaw'], 'ro', markersize=8, label='Target')
ax3.set_xlabel('time [s]')
ax3.set_ylabel('yaw [rad]')
ax3.grid(True)
ax3.legend()

plt.tight_layout()

# Save temporal profiles plot
temporal_file = output_file.replace('.png', '_temporal_profiles.png')
plt.savefig(temporal_file, dpi=300, bbox_inches='tight')
print(f"Temporal profiles plot saved to: {temporal_file}")
plt.close()

# --- PLOT 3: VELOCITY PROFILES (V, OMEGA) ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

# Linear velocity vs time
ax1.plot(t, traj_v, 'b-', linewidth=2, label='Real Trajectory')
if initial_waypoint:
    ax1.plot(0, initial_waypoint['v'], 'go', markersize=8, label='Initial')
if final_waypoint:
    ax1.plot(t_final, final_waypoint['v'], 'ro', markersize=8, label='Target')
ax1.set_ylabel('v [m/s]')
ax1.set_ylim(bottom=0)
ax1.grid(True)
ax1.legend()
ax1.set_title('Temporal Profiles: Velocities')

# Angular velocity vs time
ax2.plot(t, traj_omega, 'b-', linewidth=2, label='Real Trajectory')
if initial_waypoint:
    ax2.plot(0, initial_waypoint['omega'], 'go', markersize=8, label='Initial')
if final_waypoint:
    ax2.plot(t_final, final_waypoint['omega'], 'ro', markersize=8, label='Target')
ax2.set_xlabel('time [s]')
ax2.set_ylabel('ω [rad/s]')
ax2.grid(True)
ax2.legend()

plt.tight_layout()

# Save velocity profiles plot
velocity_file = output_file.replace('.png', '_velocity_profiles.png')
plt.savefig(velocity_file, dpi=300, bbox_inches='tight')
print(f"Velocity profiles plot saved to: {velocity_file}")
plt.close()
