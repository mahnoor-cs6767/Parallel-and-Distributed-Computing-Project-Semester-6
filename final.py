import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

# ✅ Load real data
normal_df = pd.read_csv("normal_results.csv")
ray_df = pd.read_csv("ray_results.csv")

# Merge data
df = pd.merge(normal_df, ray_df, on="workers")

# Speedup calculate
df["speedup"] = df["normal_time"] / df["ray_time"]

print("=" * 55)
print("        NORMAL vs RAY - REAL COMPARISON")
print("=" * 55)
print(df.to_string(index=False))
print("=" * 55)

workers = df["workers"]

# ✅ Graphs
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Real Performance Analysis (Auto Generated)", fontsize=14)

# 1️⃣ Time comparison
axes[0,0].plot(workers, df["normal_time"], 'ro-', label="Normal")
axes[0,0].plot(workers, df["ray_time"], 'bo-', label="Ray")
axes[0,0].set_title("Time Comparison")
axes[0,0].legend()

# 2️⃣ Speedup
axes[0,1].bar(workers, df["speedup"])
axes[0,1].set_title("Speedup")

# 3️⃣ Accuracy
axes[1,0].plot(workers, df["accuracy"], 'ro-', label="Normal")
axes[1,0].plot(workers, df["ray_accuracy"], 'bo--', label="Ray")
axes[1,0].set_title("Accuracy")
axes[1,0].legend()

# 4️⃣ Side-by-side
x = range(len(workers))
width = 0.35
axes[1,1].bar([i - width/2 for i in x], df["normal_time"], width, label="Normal")
axes[1,1].bar([i + width/2 for i in x], df["ray_time"], width, label="Ray")
axes[1,1].set_xticks(list(x))
axes[1,1].set_xticklabels(workers)
axes[1,1].set_title("Side by Side Time")
axes[1,1].legend()

plt.tight_layout()
plt.savefig("final_graph.png")
plt.show()

print("\n✅ Graph saved as final_graph.png")