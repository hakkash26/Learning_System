import os
import json
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = "outputs"
csv_path = os.path.join(OUT_DIR, "java_learning_reward_multi_seed.csv")
df = pd.read_csv(csv_path)
episodes = df["Episode"].values
mean_rewards = df["Mean Reward"].values
std_rewards = df["Standard Deviation"].values
moving_avg = df["Moving Average"].values

summary_path = os.path.join(OUT_DIR, "java_learning_summary.json")
with open(summary_path) as f:
    summary = json.load(f)
results = summary["table1"]
algos = list(results.keys())

fig, (ax_r1, ax_r2) = plt.subplots(1, 2, figsize=(13, 5))

# Subplot 1: KG - RL Multi-Seed Training Reward Curve
ax_r1.plot(episodes, mean_rewards, color="#3b82f6", linewidth=1.8, label="Mean Training Reward")
ax_r1.plot(episodes, moving_avg, color="#1d4ed8", linewidth=2.5, label="Moving Avg (w=4)")
ax_r1.fill_between(
    episodes,
    mean_rewards - std_rewards,
    mean_rewards + std_rewards,
    color="#93c5fd",
    alpha=0.35,
    label="±1 Std Dev"
)

# Epoch dividers
for ep in range(1, summary["train_epochs"]):
    n_per_epoch = len(episodes) // summary["train_epochs"]
    ax_r1.axvline(x=ep * n_per_epoch, color="#94a3b8", linestyle="--", alpha=0.6, linewidth=1)

ax_r1.set_xlabel("Training Episodes")
ax_r1.set_ylabel("Cumulative Episode Reward")
ax_r1.set_title("KG - RL Multi-Seed Training Reward Curve", fontweight="bold", fontsize=11)
ax_r1.legend(loc="lower right")
ax_r1.grid(True, linestyle="--", alpha=0.3)

# Subplot 2: Comparative Cumulative Discounted Return (G) across all algorithms
g_vals = [results[name]["G"] for name in algos]
algo_color_map = {
    "KG-RL": "#2563eb",
    "MC": "#0ea5e9",
    "KG-H": "#8b5cf6",
    "CF": "#10b981",
    "Rule": "#f59e0b"
}
bar_colors = [algo_color_map.get(name, "#3b82f6") for name in algos]
bars_g = ax_r2.bar(algos, g_vals, color=bar_colors, width=0.55, edgecolor="none")
ax_r2.bar_label(bars_g, fmt="%.3f", padding=3, fontsize=8, fontweight="bold")
ax_r2.set_xlabel("Algorithms")
ax_r2.set_ylabel("Discounted Cumulative Return (G)")
ax_r2.set_title("Test-Set Cumulative Return (G) Comparison", fontweight="bold", fontsize=11)
max_g = max(g_vals) if g_vals else 7.5
ax_r2.set_ylim(0, max_g * 1.18)
ax_r2.grid(axis='y', linestyle="--", alpha=0.3)

plt.tight_layout()
g3_path = os.path.join(OUT_DIR, "java_learning_reward_curve.png")
plt.savefig(g3_path, dpi=150)
plt.close()
print(f"Successfully generated {g3_path} with title 'KG - RL Multi-Seed Training Reward Curve'")
