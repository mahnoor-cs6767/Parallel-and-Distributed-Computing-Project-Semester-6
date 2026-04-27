import warnings
warnings.filterwarnings("ignore")   # ❌ hide warnings

import time
import ray
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

ray.init(ignore_reinit_error=True, log_to_driver=False)

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Normal training times (sequential)
def normal_train(runs):
    start = time.time()
    for _ in range(runs):
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
    return time.time() - start

# Ray training times (parallel)
@ray.remote
def ray_train():
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

def ray_train_timed(workers):
    start = time.time()
    ray.get([ray_train.remote() for _ in range(workers)])
    return time.time() - start

worker_counts = [1, 2, 4, 8]
normal_times = [normal_train(w) for w in worker_counts]
ray_times    = [ray_train_timed(w) for w in worker_counts]
speedups     = [n/r for n, r in zip(normal_times, ray_times)]

print("\n📊 Speedup Analysis:")
print(f"{'Workers':<10} {'Normal(s)':<12} {'Ray(s)':<12} {'Speedup':<10}")
for i, w in enumerate(worker_counts):
    print(f"{w:<10} {normal_times[i]:<12.3f} {ray_times[i]:<12.3f} {speedups[i]:<10.2f}x")

# Graph
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(worker_counts, normal_times, 'ro-', label='Normal')
plt.plot(worker_counts, ray_times,   'bo-', label='Ray')
plt.xlabel('Workers/Runs')
plt.ylabel('Time (seconds)')
plt.title('Normal vs Ray Time')
plt.legend()

plt.subplot(1,2,2)
plt.bar(worker_counts, speedups, color='green')
plt.xlabel('Workers')
plt.ylabel('Speedup (x times faster)')
plt.title('Ray Speedup Factor')

plt.tight_layout()
plt.savefig('speedup_graph.png')
plt.show()
print("\n✅ Graph saved as speedup_graph.png")
ray.shutdown()