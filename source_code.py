import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("dataset.csv")

# Encode categorical columns
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
df['Subscription'] = LabelEncoder().fit_transform(df['Subscription'])

# Features and target
X = df.drop(['CustomerID', 'Churn'], axis=1)
y = df['Churn']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Confusion Matrix:
", confusion_matrix(y_test, y_pred))
print("
Classification Report:
", classification_report(y_test, y_pred))