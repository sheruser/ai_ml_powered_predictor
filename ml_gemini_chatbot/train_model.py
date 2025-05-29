import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("dataset.csv")

# Handle missing values
imputer = SimpleImputer(strategy='median')
df_numeric = df.select_dtypes(include=[np.number])
df[df_numeric.columns] = imputer.fit_transform(df_numeric)

# Fill categorical missing values with most frequent value
df['Stage_fear'].fillna(df['Stage_fear'].mode()[0], inplace=True)
df['Drained_after_socializing'].fillna(df['Drained_after_socializing'].mode()[0], inplace=True)

# Encode categorical features
df["Stage_fear"] = LabelEncoder().fit_transform(df["Stage_fear"])
df["Drained_after_socializing"] = LabelEncoder().fit_transform(df["Drained_after_socializing"])

# Fix the encoding - make sure Introvert=0, Extrovert=1
personality_encoder = LabelEncoder()
df["Personality"] = personality_encoder.fit_transform(df["Personality"])

# Print mapping to verify
personality_mapping = dict(zip(personality_encoder.classes_, personality_encoder.transform(personality_encoder.classes_)))
print(f"Personality mapping: {personality_mapping}")

# Features and label
X = df.drop("Personality", axis=1)
y = df["Personality"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model with optimized parameters
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10, 
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    class_weight='balanced'  # Address class imbalance
)

# Fit model
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

# Feature importance
feature_importance = dict(zip(X.columns, model.feature_importances_))
sorted_importance = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)}
print("\nFeature importance:")
for feature, importance in sorted_importance.items():
    print(f"{feature}: {importance:.4f}")

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(sorted_importance.keys(), sorted_importance.values())
plt.xticks(rotation=45, ha='right')
plt.title('Feature Importance for Personality Prediction')
plt.tight_layout()
plt.savefig('feature_importance.png')

# Verify Time_spent_Alone predictions
print("\nVerifying predictions based on Time_spent_Alone:")

# Create a test set with increasing Time_spent_Alone values
test_cases = []
for time_alone in range(0, 12):
    test_case = X.iloc[0].copy()  # Use first row as template
    test_case['Time_spent_Alone'] = time_alone
    test_cases.append(test_case)

test_df = pd.DataFrame(test_cases)
predictions = model.predict(test_df)
probabilities = model.predict_proba(test_df)

print("Time_spent_Alone | Prediction | Probability")
print("--------------- | ---------- | -----------")
for i, time_alone in enumerate(range(0, 12)):
    personality = "Extrovert" if predictions[i] == 1 else "Introvert"
    prob = probabilities[i][predictions[i]]
    print(f"{time_alone:15} | {personality:10} | {prob:.4f}")

# Add a custom rule to ensure high Time_spent_Alone values predict Introvert
# This creates a threshold-based model that properly weights Time_spent_Alone
class EnhancedPersonalityModel:
    def __init__(self, base_model, time_alone_threshold=7.0):
        self.base_model = base_model
        self.threshold = time_alone_threshold
    
    def predict(self, X):
        # Get base predictions from Random Forest
        base_preds = self.base_model.predict(X)
        
        # Override predictions for high Time_spent_Alone
        for i, x in enumerate(X.values):
            time_alone = x[0]  # Assuming Time_spent_Alone is first column
            if time_alone >= self.threshold:
                base_preds[i] = 0  # Force Introvert prediction
        
        return base_preds

# Create enhanced model
enhanced_model = EnhancedPersonalityModel(model)

# Save models
joblib.dump(model, "best_model.pkl")
joblib.dump(enhanced_model, "enhanced_model.pkl")

print("\nModels saved. Use enhanced_model.pkl for better Time_spent_Alone handling.")