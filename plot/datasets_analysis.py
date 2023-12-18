import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import find_mean_coord, DATASETS_META, ROOT_DIR

MAX_SEQ_LEN = 2048

################################################################################
print("Preparing datasets...")

for name, meta in DATASETS_META.items():
    with open(meta["path"], "rb") as f:
        data = pickle.load(f)

    filtered = []
    for pair in data:
        input_tokens, output_tokens = pair

        if isinstance(input_tokens, int):
            input_tokens = [1000] * input_tokens
        if isinstance(output_tokens, int):
            output_tokens = [1000] * output_tokens
        
        input_len = len(input_tokens)
        output_len = len(output_tokens)

        # Filter out too long sequences.
        if input_len + output_len < MAX_SEQ_LEN:
            filtered.append((input_len, output_len))

    DATASETS_META[name]["requests"] = filtered

print("Finished.")

################################################################################
print("Plotting Prompt Input Length CDF...")

fig, ax = plt.subplots(figsize=[6.4, 4.8])

for name, meta in DATASETS_META.items():
    requests = meta["requests"]
    
    # Plot line
    N = len(requests)
    data = [req[0] for req in requests]

    x = np.sort(data)
    y = np.arange(N) / float(N)

    ax.plot(x, y, label=meta["name"], color=meta["color"], linewidth=2)

    # Plot mean values
    x_mean, y_mean = find_mean_coord(x, y)

    ax.plot(x_mean, y_mean, marker="o", markersize=5, color=meta["color"])

# Plot median line
xmin, xmax = ax.get_xlim()

ax.hlines(y=0.5, xmin=xmin, xmax=xmax, colors="#999999", ls="--", linewidth=1,
          alpha=0.8)

ax.legend(loc="lower left", bbox_to_anchor=(0., 1.02, 1., .102), fontsize=16,
          ncols=2, handlelength=2, mode="expand")

ax.set_ylabel("CDF-Probability", size=18)
ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.tick_params(axis="y", labelsize=18)

ax.set_xlabel("Prompt Input Length", size=18)
ax.set_xscale("log", base=10, subs=[])
ax.set_xticks([10, 100, 1000])
ax.set_xticklabels(["10", "100", "1000"])
ax.tick_params(axis="x", labelsize=18)

file_path = os.path.join(ROOT_DIR, "figures", "datasets_analysis",
                         "prompt_input_length.pdf")

print(f"Saving {file_path}...")
fig.savefig(file_path, dpi=300, bbox_inches="tight", format="pdf")
print("Finished.")

################################################################################
print("Plotting Generated Output Length CDF...")

fig, ax = plt.subplots(figsize=[6.4, 4.8])

for name, meta in DATASETS_META.items():
    requests = meta["requests"]
    
    # Plot line
    N = len(requests)
    data = [req[1] for req in requests]

    x = np.sort(data)
    y = np.arange(N) / float(N)

    ax.plot(x, y, label=meta["name"], color=meta["color"], linewidth=2)

    # Plot mean values
    x_mean, y_mean = find_mean_coord(x, y)

    ax.plot(x_mean, y_mean, marker="o", markersize=5, color=meta["color"])

# Plot median line
xmin, xmax = ax.get_xlim()

ax.hlines(y=0.5, xmin=xmin, xmax=xmax, colors="#999999", ls="--", linewidth=1,
          alpha=0.8)

ax.legend(loc="lower left", bbox_to_anchor=(0., 1.02, 1., .102), fontsize=16,
          ncols=2, handlelength=2, mode="expand")

ax.set_ylabel("CDF-Probability", size=18)
ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.tick_params(axis="y", labelsize=18)

ax.set_xlabel("Generated Output Length", size=18)
ax.set_xscale("log", base=10, subs=[])
ax.set_xticks([1, 10, 100, 1000])
ax.set_xticklabels(["1", "10", "100", "1000"])
ax.tick_params(axis="x", labelsize=18)

file_path = os.path.join(ROOT_DIR, "figures", "datasets_analysis",
                         "generated_output_length.pdf")

print(f"Saving {file_path}...")
fig.savefig(file_path, dpi=300, bbox_inches="tight", format="pdf")
print("Finished.")

################################################################################
logbins = []
for i in [1, 2, 3, 4, 5, 6, 7, 8 ,9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
          200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000]:
    logbins.append(np.log10(i))

print("Plotting Prompt Input Length KDE...")

plt.rcdefaults()
fig, ax = plt.subplots(figsize=[6.4, 4.8])

for name, meta in DATASETS_META.items():
    requests = meta["requests"]

    data = [req[0] for req in requests]

    sns.histplot(ax=ax, data=data, color=meta["color"], log_scale=True,
                 kde=True, stat="probability", bins=logbins)

    ax.get_lines()[-1].set_label(meta["name"])
    ax.get_lines()[-1].set_linewidth(2)
    ax.containers[0].remove() 
    ax.relim()
    ax.autoscale_view()

ax.legend(loc="lower left", bbox_to_anchor=(0., 1.02, 1., .102), fontsize=16,
          ncols=2, handlelength=2, mode="expand")

ax.set_ylabel("Probability", size=18)
ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.set_yticklabels(["0%", "10%", "20%", "30%", "40%", "50%"])
ax.tick_params(axis="y", labelsize=18)


ax.set_xlabel("Prompt Input Length", size=18)
ax.set_xscale("log", base=10, subs=[])
ax.set_xticks([1, 10, 100, 1000])
ax.set_xticklabels(["1", "10", "100", "1000"])
ax.tick_params(axis="x", labelsize=18)

file_path = os.path.join(ROOT_DIR, "figures", "datasets_analysis",
                         "prompt_input_length_kde.pdf")

print(f"Saving {file_path}...")
fig.savefig(file_path, dpi=300, bbox_inches="tight", format="pdf")
print("Finished.")

################################################################################
print("Plotting Generated Output Length KDE...")

plt.rcdefaults()
fig, ax = plt.subplots(figsize=[6.4, 4.8])

for name, meta in DATASETS_META.items():
    requests = meta["requests"]

    data = [req[1] for req in requests]

    sns.histplot(ax=ax, data=data, color=meta["color"], log_scale=True,
                 kde=True, stat="probability", bins=logbins)

    ax.get_lines()[-1].set_label(meta["name"])
    ax.get_lines()[-1].set_linewidth(2)
    ax.containers[0].remove() 
    ax.relim()
    ax.autoscale_view()

ax.legend(loc="lower left", bbox_to_anchor=(0., 1.02, 1., .102), fontsize=16,
          ncols=2, handlelength=2, mode="expand")

ax.set_ylabel("Probability", size=18)
ax.set_yticks([0.0, 0.1, 0.2, 0.3])
ax.set_yticklabels(["0%", "10%", "20%", "30%"])
ax.tick_params(axis="y", labelsize=18)


ax.set_xlabel("Generated Output Length", size=18)
ax.set_xscale("log", base=10, subs=[])
ax.set_xticks([1, 10, 100, 1000])
ax.set_xticklabels(["1", "10", "100", "1000"])
ax.tick_params(axis="x", labelsize=18)

file_path = os.path.join(ROOT_DIR, "figures", "datasets_analysis",
                         "generated_output_length_kde.pdf")

print(f"Saving {file_path}...")
fig.savefig(file_path, dpi=300, bbox_inches="tight", format="pdf")
print("Finished.")

################################################################################
print("Plotting KDEs per Dataset...")

for name, meta in DATASETS_META.items():
    requests = meta["requests"]
    
    data = [req[0] for req in requests]

    fig, ax = plt.subplots(figsize=[6.4, 4.8])

    sns.histplot(ax=ax, data=data, color=meta["color"], label=meta["name"],
                 linewidth=2, log_scale=True, kde=True, stat="probability", 
                 bins=logbins)

    ax.legend(loc="upper right", fontsize=16, ncols=1, handlelength=2)

    ax.set_ylabel("Probability", size=18)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax.set_yticklabels(["0%", "10%", "20%", "30%", "40%", "50%", "60%"])
    ax.tick_params(axis="y", labelsize=18)
    ax.set_ylim(top=0.65)


    ax.set_xlabel("Prompt Input Length", size=18)
    ax.set_xscale("log", base=10, subs=[])
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1000"])
    ax.tick_params(axis="x", labelsize=18)

    file_path = os.path.join(ROOT_DIR, "figures", "datasets_analysis",
                             f"prompt_input_length_kde_{name}.pdf")
    
    print(f"Saving {file_path}...")
    fig.savefig(file_path, dpi=300, bbox_inches="tight", format="pdf")
    print("Finished.")

for name, meta in DATASETS_META.items():
    requests = meta["requests"]
    
    data = [req[1] for req in requests]

    fig, ax = plt.subplots(figsize=[6.4, 4.8])

    sns.histplot(ax=ax, data=data, color=meta["color"], label=meta["name"],
                 linewidth=1, log_scale=True, kde=True, stat="probability",
                 bins=logbins)

    ax.legend(loc="upper right", fontsize=16, ncols=1, handlelength=2)

    ax.set_ylabel("Probability", size=18)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3])
    ax.set_yticklabels(["0%", "10%", "20%", "30%"])
    ax.tick_params(axis="y", labelsize=18)
    ax.set_ylim(top=0.35)


    ax.set_xlabel("Generated Output Length", size=18)
    ax.set_xscale("log", base=10, subs=[])
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1000"])
    ax.tick_params(axis="x", labelsize=18)

    file_path = os.path.join(ROOT_DIR, "figures", "datasets_analysis",
                             f"generated_output_length_kde_{name}.pdf")
    
    print(f"Saving {file_path}...")
    fig.savefig(file_path, dpi=300, bbox_inches="tight", format="pdf")
    print("Finished.")

# python plot/datasets_analysis.py