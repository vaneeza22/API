import joblib
from sklearn.linear_model import LogisticRegression

# 1. Provide sample data or load your dataset
X = [[2.0, 50], [5.0, 75], [1.0, 30], [8.0, 90], [3.5, 60], [7.0, 85]]
y = [0, 1, 0, 1, 0, 1]  # 0 = Fail, 1 = Pass

# 2. Initialize and train the model
print("Training the model...")
model = LogisticRegression()
model.fit(X, y)

# 3. Save the trained model to disk (this creates model.pkl)
joblib.dump(model, "model.pkl")
print("Success! Model saved as 'model.pkl'.")