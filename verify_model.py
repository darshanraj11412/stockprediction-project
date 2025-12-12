import os

model_path = os.path.join(os.path.dirname(__file__), 'stock_model.joblib')
print(f"Looking for model at: {model_path}")
print("Exists?", os.path.isfile(model_path))

