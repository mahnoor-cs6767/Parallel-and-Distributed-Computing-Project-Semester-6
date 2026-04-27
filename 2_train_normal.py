import time
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("🚀 Normal Training Start...")

data = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

runs = [1, 2, 4, 8]

normal_times = []
accuracies = []

for r in runs:
    start = time.time()
    acc_list = []

    for _ in range(r):
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        acc_list.append(model.score(X_test, y_test))

    end = time.time()

    avg_acc = sum(acc_list) / len(acc_list)
    total_time = end - start

    normal_times.append(total_time)
    accuracies.append(avg_acc)

    print(f"\nRuns: {r}")
    print("Accuracy:", avg_acc)
    print("Time:", total_time)

# ✅ Save results
df = pd.DataFrame({
    "workers": runs,
    "normal_time": normal_times,
    "accuracy": accuracies
})

df.to_csv("normal_results.csv", index=False)
print("\n✅ normal_results.csv saved!")