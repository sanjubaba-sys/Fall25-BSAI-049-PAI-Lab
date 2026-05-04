# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# 1️⃣ Load dataset
train = pd.read_csv("trainfile.csv")
test = pd.read_csv("testfile.csv")

print("Train Shape:", train.shape)
print("Test Shape:", test.shape)

# Save PassengerId for submission
passenger_ids = test["PassengerId"]

# 2️⃣ EDA
print("\nDataset Info:")
print(train.info())

print("\nMissing Values:")
print(train.isnull().sum())

# 3️⃣ Data Preprocessing

# Numeric columns
numeric_cols = ["Age","RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]

for col in numeric_cols:
    train[col] = train[col].fillna(train[col].median())
    test[col] = test[col].fillna(train[col].median())

# Categorical columns
categorical_cols = ["HomePlanet","CryoSleep","Destination","VIP"]

for col in categorical_cols:
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(train[col].mode()[0])

# Convert boolean to int
train["CryoSleep"] = train["CryoSleep"].astype(int)
train["VIP"] = train["VIP"].astype(int)
train["Transported"] = train["Transported"].astype(int)

test["CryoSleep"] = test["CryoSleep"].astype(int)
test["VIP"] = test["VIP"].astype(int)

# Encode categorical columns
encoder = LabelEncoder()

train["HomePlanet"] = encoder.fit_transform(train["HomePlanet"])
test["HomePlanet"] = encoder.transform(test["HomePlanet"])

train["Destination"] = encoder.fit_transform(train["Destination"])
test["Destination"] = encoder.transform(test["Destination"])

# Drop unnecessary columns
train = train.drop(["PassengerId","Name","Cabin"], axis=1)
test = test.drop(["PassengerId","Name","Cabin"], axis=1)

# 4️⃣ Split dataset
X = train.drop("Transported", axis=1)
y = train["Transported"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5️⃣ Train model
model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
model.fit(X_train, y_train)

# 6️⃣ Evaluate model
predictions = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

# 7️⃣ Predict test data
test_preds = model.predict(test)

# Convert 0/1 to True/False (important for Kaggle)
test_preds = test_preds.astype(bool)

# 8️⃣ Create submission file
submission = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Transported": test_preds
})

submission.to_csv("submission.csv", index=False)

print("\nSubmission file created successfully!")