import pandas as pd
from train import train_and_evaluate_model

df = pd.read_csv('df5.csv')

model, metrics = train_and_evaluate_model(df)

print("\nModel Performance Metrics:")
print(f"RMSE: {metrics['RMSE']:.2f} seconds")
print(f"R2 Score: {metrics['R2']:.3f}")
print(f"Test Set Size: {metrics['Test_Size']} incidents")
