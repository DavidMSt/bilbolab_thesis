import os
import numpy as np
import matplotlib.pyplot as plt

# ================================
# Parameters you will likely tweak
# ================================
output_dir = "frames"  # where PNGs go
fps = 30  # frames per second (match your video)
duration = 5.0  # seconds
n_frames = int(fps * duration)

plt.rcParams.update({
    "font.size": 28,  # larger base text
    "axes.titlesize": 32,
    "axes.labelsize": 28,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "axes.linewidth": 2.5,  # thicker axes/frame
    "lines.linewidth": 4,  # default line thickness
})

# Match your video aspect ratio: 16:9, 1920x1080-ish
fig_w, fig_h, dpi = 16, 9, 120  # gives 1920x1080 pixels

# "Experiment" time axis (fixed)
t_start, t_end = 0.0, duration
t_full = np.linspace(t_start, t_end, 500)  # full-resolution time grid
y_full = np.sin(2 * np.pi * t_full / duration)  # some progression

# Create output folder
os.makedirs(output_dir, exist_ok=True)

# =========================
# Set up the static figure
# =========================
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

# Transparent figure background
fig.patch.set_alpha(0.0)
ax.set_facecolor("none")

# Dark-theme styling for a grey/black video background
ax.spines["bottom"].set_color("white")
ax.spines["top"].set_color("white")
ax.spines["left"].set_color("white")
ax.spines["right"].set_color("white")

ax.tick_params(axis="x", colors="white")
ax.tick_params(axis="y", colors="white")

ax.yaxis.label.set_color("white")
ax.xaxis.label.set_color("white")
ax.title.set_color("white")

# Optional: dim grid in white
ax.grid(True, alpha=0.4, color="white", linestyle=':', linewidth=4)

# Fix axes once (no rescaling)
ax.set_xlim(t_start, t_end)
ax.set_ylim(y_full.min() - 0.2, y_full.max() + 0.2)

ax.set_xlabel("Time [s]")
ax.set_ylabel("Signal")
ax.set_title("Example Progression vs Time")

# Line objects (one for the trail, one for a moving marker)
(line,) = ax.plot([], [], lw=6, color="white")
(marker,) = ax.plot([], [], "o", color="cyan", markersize=12)

# =========================
# Generate each frame
# =========================
for f in range(n_frames):
    # Fraction of the total time covered at this frame
    frac = (f + 1) / n_frames  # (0, 1]
    t_max = t_start + frac * (t_end - t_start)

    # Mask data up to t_max
    mask = t_full <= t_max
    t_vis = t_full[mask]
    y_vis = y_full[mask]

    # Update the line (trail)
    line.set_data(t_vis, y_vis)

    # Update the moving marker at the last visible point
    if len(t_vis) > 0:
        marker.set_data([t_vis[-1]], [y_vis[-1]])

    # Save current frame as PNG with transparency
    frame_path = os.path.join(output_dir, f"frame_{f:05d}.png")
    plt.savefig(
        frame_path,
        transparent=True,  # <- alpha actually goes into PNG
        bbox_inches="tight",
        pad_inches=0
    )

plt.close(fig)
print(f"Saved {n_frames} frames in '{output_dir}/'")
