
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load both datasets and combine them
df1 = pd.read_csv("dataset\expanded_transactions_1.csv")
df2 = pd.read_csv("dataset\expanded_transactions_2.csv")
df = pd.concat([df1, df2], ignore_index=True)
print("Dataset Loaded Successfully!")
print(df.head())  # Show first few rows of the dataset
print(df.columns)  # Show all column names
print("Total Transactions:", len(df))  # Show dataset size
encoder = LabelEncoder()
df["Sender UPI ID"] = encoder.fit_transform(df["Sender UPI ID"])
df["Receiver UPI ID"] = encoder.fit_transform(df["Receiver UPI ID"])
df["Status"] = df["Status"].map({"SUCCESS": 0, "FAILED": 1})  # Convert status to binary

# Select Features and Target
features = ["Amount (INR)", "Sender UPI ID", "Receiver UPI ID", "Status"]
X = df[features]  # Inputs
y = df["is_fraud"]  # Target (Fraud or Not)

# ✅ Check if feature selection worked
print("Feature Selection Done!")
print(X.head())  # Show first few rows of X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Check if data split correctly
print("Training Data Size:", len(X_train))
print("Testing Data Size:", len(X_test))
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_split=5,
    random_state=42
)

model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Save Model
joblib.dump(model, "model/fraud_detection_model.pkl")
print("✅ Model Saved Successfully!")
