"""
NeuroPlay 2.0 - Chart generation for the progress report.
Each function saves a PNG to out_dir and returns its path, ready to
embed into the ReportLab PDF.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def trajectory_quiver_chart(df, out_dir):
    """X-Y trajectory as a quiver plot: arrows show direction, color = speed."""
    fig, ax = plt.subplots(figsize=(6, 5))
    x, y = df["player_x"].values, df["player_y"].values
    u, v = np.diff(x, append=x[-1]), np.diff(y, append=y[-1])
    speed = df["speed"].values if "speed" in df.columns else np.ones(len(x))

    q = ax.quiver(
        x, y, u, v, speed, angles="xy", scale_units="xy", scale=1,
        cmap="plasma", width=0.004,
    )
    fig.colorbar(q, ax=ax, label="Speed")
    ax.set_title("Movement Trajectory & Direction")
    ax.set_xlabel("Player X (from roll)")
    ax.set_ylabel("Player Y (from pitch)")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()

    path = os.path.join(out_dir, "trajectory.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def range_of_motion_radar(motion_metrics, out_dir):
    """Polar chart summarizing pitch/roll extremes as a single glance view."""
    pitch_min, pitch_max = motion_metrics["pitch_range"]
    roll_min, roll_max = motion_metrics["roll_range"]

    labels = ["Pitch +", "Roll +", "Pitch -", "Roll -"]
    values = [max(pitch_max, 0), max(roll_max, 0), abs(min(pitch_min, 0)), abs(min(roll_min, 0))]
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "polar"})
    ax.plot(angles, values, "o-", linewidth=2, color="#2a6fdb")
    ax.fill(angles, values, alpha=0.25, color="#2a6fdb")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("Range of Motion", pad=20)
    fig.tight_layout()

    path = os.path.join(out_dir, "range_of_motion.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def score_vs_time_chart(df, score_metrics, out_dir):
    """Score progression with vertical markers at each speed-tier change."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(df["t_sec"], df["score"], color="#1a7a3c", linewidth=2)

    for speed_val, t in score_metrics["speed_tier_times"].items():
        if t > 0:
            ax.axvline(t, color="gray", linestyle="--", alpha=0.6)
            ax.text(t, ax.get_ylim()[1] * 0.95, f"{speed_val}x", fontsize=8, rotation=90,
                    va="top", ha="right", color="gray")

    ax.set_title("Score Over Time (dashed lines = speed tier change)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Score")
    fig.tight_layout()

    path = os.path.join(out_dir, "score_vs_time.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def emg_threshold_chart(df, emg_threshold, out_dir):
    """EMG signal with threshold line and shaded contraction windows."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(df["t_sec"], df["emg_value"], color="#b33939", linewidth=1)
    ax.axhline(emg_threshold, color="black", linestyle="--", linewidth=1, label="Threshold")

    contracted = df["emg_value"] >= emg_threshold
    ax.fill_between(df["t_sec"], 0, df["emg_value"].max(), where=contracted,
                     color="orange", alpha=0.15, step="mid")

    ax.set_title("EMG Signal & Contraction Windows")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EMG Value")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    path = os.path.join(out_dir, "emg_threshold.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def kinematics_stack_chart(df, ldlj_windows, out_dir):
    """Speed / acceleration / smoothness (LDLJ) stacked over time.
    LDLJ replaces raw jerk here - raw jerk is too noisy to read a trend
    from and isn't used on its own in the stroke kinematics literature;
    LDLJ is the validated smoothness metric (closer to 0 = smoother)."""
    fig, axes = plt.subplots(3, 1, figsize=(7, 6.5), sharex=True)

    axes[0].plot(df["t_sec"], df["speed"], color="#2a6fdb", linewidth=1)
    axes[0].set_ylabel("Speed")
    axes[0].grid(alpha=0.2)

    axes[1].plot(df["t_sec"], df["accel"], color="#e08e0b", linewidth=1)
    axes[1].set_ylabel("Acceleration")
    axes[1].grid(alpha=0.2)

    if len(ldlj_windows):
        axes[2].plot(ldlj_windows["t_center"], ldlj_windows["ldlj"],
                     color="#8e44ad", linewidth=1.5, marker="o", markersize=3)
    axes[2].set_ylabel("Smoothness\n(LDLJ)")
    axes[2].grid(alpha=0.2)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Movement Kinematics")
    fig.tight_layout()

    path = os.path.join(out_dir, "kinematics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_all_charts(df, metrics, out_dir, emg_threshold=400.0):
    os.makedirs(out_dir, exist_ok=True)
    return {
        "trajectory": trajectory_quiver_chart(df, out_dir),
        "range_of_motion": range_of_motion_radar(metrics["motion"], out_dir),
        "score_vs_time": score_vs_time_chart(df, metrics["score"], out_dir),
        "emg_threshold": emg_threshold_chart(df, emg_threshold, out_dir),
        "kinematics": kinematics_stack_chart(df, metrics["motion"]["ldlj_windows"], out_dir),
    }
