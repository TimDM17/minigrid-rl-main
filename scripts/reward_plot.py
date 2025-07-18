

import re
import matplotlib.pyplot as plt

# Paths to the log files
log_file_path1 = "/Users/yezihan/Desktop/minigrid/rl-starter-files_roomSetting/scripts/storage/adaptive_ppo_reasoner_16*16_seed6./log.txt"
log_file_path2 = "/Users/yezihan/Desktop/minigrid/rl-starter-files_roomSetting/scripts/storage/adaptive_ppo_reasoner_16*16_seed1./log.txt"

# Function to parse the log file and extract frames and mean returns for every 30 updates
def parse_log_file(file_path):
    frames = []
    mean_returns = []
    log_pattern = re.compile(r"U (\d+) \| F (\d+) .* rR:μσmM ([\d.-]+)")
    with open(file_path, "r") as file:
        for line in file:
            match = log_pattern.search(line)
            if match:
                update = int(match.group(1))  # Extract update number
                if update % 50 == 0:  # Include only updates that are multiples of 30
                    frames.append(int(match.group(2)))  # Extract frames
                    mean_returns.append(float(match.group(3)))  # Extract mean return
    return frames, mean_returns

# Parse both log files
frames1, mean_returns1 = parse_log_file(log_file_path1)
frames2, mean_returns2 = parse_log_file(log_file_path2)

# Plot return vs. frames for both logs
plt.figure(figsize=(10, 6))

# Plot the first log
plt.plot(frames1, mean_returns1, label="Without Dylan", linewidth=2, color='royalblue')

# Plot the second log
plt.plot(frames2, mean_returns2, label="With Dylan", linewidth=2, color='indianred')

# Add labels, title, legend, and grid
plt.xlabel("Frames (Steps)", fontsize=18)
plt.ylabel("Mean Return", fontsize=18)
plt.title("Return over Steps", fontsize=20)
plt.legend()
# Update legend with larger font size
plt.legend(fontsize=18)  # Set font size for the legend
plt.grid(True)
plt.tight_layout()

# Save the plot or display it
#plt.savefig("return_comparison.png")  # Save the plot as a PNG file
plt.show()  # Display the plot

