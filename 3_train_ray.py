import warnings
warnings.filterwarnings("ignore")

import ray
import time
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("⚡ Ray Distributed Training Start...")

if not ray.is_initialized():
    ray.init(ignore_reinit_error=True, log_to_driver=False)

data = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

@ray.remote
def train():
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

workers = [1, 2, 4, 8]

ray_times = []
accuracies = []

for w in workers:
    start = time.time()

    results = ray.get([train.remote() for _ in range(w)])

    end = time.time()

    avg_acc = sum(results) / len(results)
    total_time = end - start

    ray_times.append(total_time)
    accuracies.append(avg_acc)

    print(f"\nWorkers: {w}")
    print("Accuracy:", avg_acc)
    print("Time:", total_time)

# ✅ Save results
df = pd.DataFrame({
    "workers": workers,
    "ray_time": ray_times,
    "ray_accuracy": accuracies
})

df.to_csv("ray_results.csv", index=False)
print("\n✅ ray_results.csv saved!")

ray.shutdown()