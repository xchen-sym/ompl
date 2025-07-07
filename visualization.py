#!/usr/bin/env python3
import re
import math
import matplotlib.pyplot as plt

dt = 0.01


def parse_log(filename):
    """
    Parse an SST_run.log file to extract segments of:
      (x, y, yaw, v, omega, a, alpha, num_steps)
    """
    segments = []
    with open(filename, 'r') as f:
        lines = [line.rstrip() for line in f]

    for i, line in enumerate(lines):
        if line.startswith("At state Compound state"):
            xy_line   = lines[i+1].strip()    # RealVectorState [x y]
            yaw_line   = lines[i+2].strip()    # SO2State [yaw]
            vel_line   = lines[i+4].strip()    # RealVectorState [v omega]
            steps_line = lines[i+5].strip()    # for N steps

            # Extract x, y
            x_str, y_str = re.search(r"RealVectorState \[([^\]]+)\]", xy_line).group(1).split()
            x, y = float(x_str), float(y_str)

            # Extract yaw
            yaw = float(re.search(r"SO2State \[([^\]]+)\]", yaw_line).group(1))

            # Extract v, omega
            v_str, omega_str = re.search(r"RealVectorControl \[([^\]]+)\]", vel_line).group(1).split()
            v, omega = float(v_str), float(omega_str)

            # Extract step count
            steps = int(re.search(r"for (\d+) steps", steps_line).group(1))

            segments.append((x, y, yaw, v, omega, steps))

        elif line.startswith("Arrive at state Compound state"):
            xy_line_final   = lines[i+1].strip()    # Final RealVectorState [x y]
            yaw_line_final   = lines[i+2].strip()    # Final SO2State [yaw]

            # Extract x_final, y_final
            x_str_final, y_str_final = re.search(r"RealVectorState \[([^\]]+)\]", xy_line_final).group(1).split()
            x_final, y_final = float(x_str_final), float(y_str_final)

            # Extract yaw_final
            yaw_final = float(re.search(r"SO2State \[([^\]]+)\]", yaw_line_final).group(1))

            # Keep last step's v and omega as a_final and alpha_final
            _, _, _, v_final, omega_final, _ = segments[-1]

            steps_final = 1
            segments.append((x_final, y_final, yaw_final, v_final, omega_final, steps_final))
    
    return segments


def plotting(segments):
    x_list, y_list, yaw_list = [], [], []
    v_list, omega_list = [], []

    for x, y, yaw, v, omega, steps in segments:
        for _ in range(steps):
            x_list.append(x)
            y_list.append(y)
            yaw_list.append(yaw)
            v_list.append(v)
            omega_list.append(omega)

            # propagation
            x += v * dt * math.cos(yaw)
            y += v * dt * math.sin(yaw)
            yaw += omega * dt

    t_list = [0.01 * i for i in range(len(x_list))]

    # Spatial profile
    plt.figure()
    plt.plot(x_list, y_list)
    plt.xlabel("x", fontsize=20)
    plt.ylabel("y", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title("Spatial Profile", fontsize=30, pad=20)
    plt.grid(True)
    plt.show()

    # Temporal profiles
    plt.figure()
    plt.plot(t_list, x_list, label='x')
    plt.plot(t_list, y_list, label='y')
    plt.plot(t_list, yaw_list, label='yaw')
    plt.xlabel('t', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Temporal Profiles', fontsize=30, pad=20)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.show()

    # Velocity profiles
    plt.figure()
    plt.plot(t_list, v_list, label='v')
    plt.plot(t_list, omega_list, label=r'$\omega$')
    plt.xlabel('t', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Velocity Profiles', fontsize=30, pad=20)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.show()


def main():
    log_file = 'logs/test_no_acc_constraints.log'  # path to your log file
    segments = parse_log(log_file)

    total_steps = sum(seg[-1] for seg in segments[:-1])
    total_time = total_steps * dt
    print(f"Computed {total_steps} total steps")
    print(f"Total trajectory time: {total_time:.2f} seconds")

    plotting(segments)


if __name__ == "__main__":
    main()
