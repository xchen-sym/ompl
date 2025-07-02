#!/usr/bin/env python3
import re
import math
import matplotlib.pyplot as plt


def parse_log(filename):
    """
    Parse an SST_run.log file to extract segments of:
      (x0, y0, yaw0, v, omega, num_steps)
    """
    segments = []
    with open(filename, 'r') as f:
        lines = [line.rstrip() for line in f]

    for i, line in enumerate(lines):
        if line.startswith("At state Compound state"):
            rv_line    = lines[i+1].strip()  # RealVectorState [x y]
            so2_line   = lines[i+2].strip()  # SO2State [yaw]
            ctrl_line  = lines[i+4].strip()  # apply control RealVectorControl [v omega]
            steps_line = lines[i+5].strip()  # for N steps

            # Extract x, y
            x_str, y_str = re.search(r"RealVectorState \[([^\]]+)\]", rv_line).group(1).split()
            x0, y0 = float(x_str), float(y_str)

            # Extract yaw
            yaw0 = float(re.search(r"SO2State \[([^\]]+)\]", so2_line).group(1))

            # Extract v, omega
            v_str, omega_str = re.search(r"RealVectorControl \[([^\]]+)\]", ctrl_line).group(1).split()
            v, omega = float(v_str), float(omega_str)

            # Extract step count
            steps = int(re.search(r"for (\d+) steps", steps_line).group(1))

            segments.append((x0, y0, yaw0, v, omega, steps))

    return segments


def simulate(segments, dt=0.01):
    """
    Simulate each segment forward in time.
    Returns five lists: x_list, y_list, yaw_list, v_list, omega_list.
    """
    x_list, y_list, yaw_list = [], [], []
    v_list, omega_list       = [], []
    last_v = last_omega = None

    for x0, y0, yaw0, v, omega, steps in segments:
        x, y, yaw = x0, y0, yaw0
        last_v, last_omega = v, omega
        for _ in range(steps):
            x_list.append(x)
            y_list.append(y)
            yaw_list.append(yaw)
            v_list.append(v)
            omega_list.append(omega)

            # propagate
            x   += v     * dt * math.cos(yaw)
            y   += v     * dt * math.sin(yaw)
            yaw += omega * dt

    # append final state after the very last update
    if last_v is not None:
        x_list.append(x)
        y_list.append(y)
        yaw_list.append(yaw)
        v_list.append(last_v)
        omega_list.append(last_omega)

    return x_list, y_list, yaw_list, v_list, omega_list


def main():
    log_file = 'logs/test.log'  # path to your log
    segments = parse_log(log_file)
    x_list, y_list, yaw_list, v_list, omega_list = simulate(segments)
    t_list = [0.01*i for i in range(len(x_list))]

    print(f"Computed {len(x_list)} total steps")
    # Now x_list, y_list, yaw_list, v_list, omega_list are available for use.
    # For example, to peek at the first 5 steps:
    for i in range(min(10, len(x_list))):
        print(f"Step {i}: x={x_list[i]:.4f}, y={y_list[i]:.4f}, "
              f"yaw={yaw_list[i]:.4f}, v={v_list[i]:.4f}, ω={omega_list[i]:.4f}")

    print(f"Total trajectory time: {t_list[-1]:.2f} seconds")

    # Plot spatial profile
    plt.figure()
    plt.plot(x_list, y_list)
    plt.xlabel("x", fontsize=20)
    plt.ylabel("y", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title("SST Spatial Profile", fontsize=30, pad=20)
    plt.grid(True)
    plt.show()

    # Plot temporal profiles
    plt.figure()
    plt.plot(t_list, x_list, label='x')
    plt.plot(t_list, y_list, label='y')
    plt.plot(t_list, yaw_list, label='yaw')
    plt.xlabel('t', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('SST Temporal Profiles', fontsize=30, pad=20)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.show()

    # Plot linear velocity profile
    plt.figure()
    plt.plot(t_list, v_list)
    plt.xlabel('t', fontsize=20)
    plt.ylabel('v', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('SST Linear Velocity Profile', fontsize=30, pad=20)
    plt.grid(True)
    plt.show()

    # Plot angular velocity profile
    plt.figure()
    plt.plot(t_list, omega_list)
    plt.xlabel('t', fontsize=20)
    plt.ylabel('$\omega$', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('SST Angular Velocity Profile', fontsize=30, pad=20)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
