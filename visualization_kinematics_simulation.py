import re
import matplotlib.pyplot as plt

# --- PARAMETERS ---
LOG_FILE = 'logs/simulation_no_alpha_max.log' # change log file name when needed

# --- DATA CONTAINERS ---
path = []
traj = []

# --- READ AND PARSE ---
with open(LOG_FILE, 'r') as f:
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

# --- SLICE TRAJECTORY ---
N = len(path)
traj_N = traj[:N]

# --- EXTRACT COORDINATES ---
path_x, path_y = zip(*path)
traj_x, traj_y = zip(*traj_N)

# --- PLOT ---
plt.figure(figsize=(6,6))
plt.plot(path_x, path_y, 'o-', label='Path')
plt.plot(traj_x, traj_y, 'x-', label=f'Trajectory (first {N} pts)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Path vs. Trajectory')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()
