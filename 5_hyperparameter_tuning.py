import warnings
warnings.filterwarnings("ignore")

import ray
import time
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

print("🔍 Hyperparameter Tuning (Normal vs Ray)")

# Initialize Ray
if not ray.is_initialized():
    ray.init(ignore_reinit_error=True, log_to_driver=False)

# Load dataset
data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Parameter grid
param_grid = [
    {'n_estimators': n, 'max_depth': d, 'min_samples_split': s}
    for n in [10, 50, 100]
    for d in [2, 5, None]
    for s in [2, 5]
]

# -------------------------------
# 🔹 Normal (Sequential Tuning)
# -------------------------------
def normal_tuning():
    start = time.time()

    best_score = 0
    best_params = None

    for params in param_grid:
        model = RandomForestClassifier(**params, random_state=42)
        score = cross_val_score(model, X_train, y_train, cv=3).mean()

        if score > best_score:
            best_score = score
            best_params = params

    end = time.time()

    return end - start, best_score, best_params


# -------------------------------
# ⚡ Ray (Parallel Tuning)
# -------------------------------
@ray.remote
def ray_evaluate(params):
    model = RandomForestClassifier(**params, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=3).mean()
    return score, params


def ray_tuning():
    start = time.time()

    futures = [ray_evaluate.remote(p) for p in param_grid]
    results = ray.get(futures)

    best_score, best_params = max(results, key=lambda x: x[0])

    end = time.time()

    return end - start, best_score, best_params


# -------------------------------
# 🚀 Run both methods
# -------------------------------
print("\n🔹 Running Normal Hyperparameter Tuning...")
n_time, n_score, n_params = normal_tuning()

print(f"✅ Normal Time: {n_time:.2f} sec")
print(f"✅ Normal Best Accuracy: {n_score:.4f}")
print(f"✅ Normal Best Params: {n_params}")

print("\n⚡ Running Ray Parallel Hyperparameter Tuning...")
r_time, r_score, r_params = ray_tuning()

print(f"✅ Ray Time: {r_time:.2f} sec")
print(f"✅ Ray Best Accuracy: {r_score:.4f}")
print(f"✅ Ray Best Params: {r_params}")

# -------------------------------
# 📊 Speedup
# -------------------------------
speedup = n_time / r_time

print(f"\n🚀 Speedup with Ray: {speedup:.2f}x")

# -------------------------------
# 💾 Save Results (IMPORTANT)
# -------------------------------
df = pd.DataFrame({
    "Method": ["Normal", "Ray"],
    "Time (seconds)": [n_time, r_time],
    "Best Accuracy": [n_score, r_score],
    "Best Params": [str(n_params), str(r_params)]
})

df.to_csv("tuning_results.csv", index=False)

print("\n✅ Results saved as tuning_results.csv")

# Shutdown Ray
ray.shutdown()