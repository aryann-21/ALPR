import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("runs/detect/train4/results.csv")

plt.figure(figsize=(10,6))

plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@50')
plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@50-95')

plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("YOLOv8 Model Performance")
plt.legend()
plt.grid(True)

plt.savefig("runs/detect/train4/performance_graph.png", dpi=300)
plt.show()