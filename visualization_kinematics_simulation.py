import re
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
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

# --- READ AND PARSE ---
with open(args.log_file, 'r') as f:
    in_path = False
    in_traj = False
    for line in f:
        line = line.strip()
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
            # first two entries are x, y
            traj.append((nums[0], nums[1]))

# --- EXTRACT COORDINATES ---
path_x, path_y = zip(*path)
traj_x, traj_y = zip(*traj)

# --- PLOT ---
plt.figure(figsize=(6,6))
plt.plot(path_x, path_y, 'o-', label=f'Reference Trajectory: {0.01 * (len(path) - 1):.2f}s')
plt.plot(traj_x, traj_y, 'x-', label=f'Real Trajectory: {0.01 * (len(traj) - 1):.2f}s')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Reference vs. Real Trajectory')
plt.axis('equal')
plt.grid(True)
plt.legend()

# Save plot to figures directory
output_file = args.output
output_dir = os.path.dirname(output_file)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_file}")
plt.close()  # Close the figure to free memory
