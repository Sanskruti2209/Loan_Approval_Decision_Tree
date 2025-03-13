import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Define the absolute path to the training data
TRAIN_DATA_PATH = r"C:\Users\ACER\Downloads\archive (5)\train_u6lujuX_CVtuZ9i.csv"

# Set working directory
os.chdir(r"C:\Users\ACER\OneDrive\Desktop\PGM_PROJECT")
print(f"Current working directory: {os.getcwd()}")

# Verify the training file exists
if not os.path.exists(TRAIN_DATA_PATH):
    raise FileNotFoundError(f"Training data not found at: {TRAIN_DATA_PATH}")

# Load the training dataset
print("Loading training data...")
train_data = pd.read_csv(TRAIN_DATA_PATH)
print("Data loaded successfully. First few rows:")
print(train_data.head())

# Handle missing values
train_data['Gender'].fillna(train_data['Gender'].mode()[0], inplace=True)
train_data['Married'].fillna(train_data['Married'].mode()[0], inplace=True)
train_data['Dependents'].fillna(train_data['Dependents'].mode()[0], inplace=True)
train_data['Self_Employed'].fillna(train_data['Self_Employed'].mode()[0], inplace=True)
train_data['LoanAmount'].fillna(train_data['LoanAmount'].median(), inplace=True)
train_data['Loan_Amount_Term'].fillna(train_data['Loan_Amount_Term'].mode()[0], inplace=True)
train_data['Credit_History'].fillna(train_data['Credit_History'].mode()[0], inplace=True)

# Encode categorical variables
le = LabelEncoder()
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
for col in categorical_cols:
    train_data[col] = le.fit_transform(train_data[col])

# Define features and target
X = train_data.drop(columns=['Loan_ID', 'Loan_Status'])
y = train_data['Loan_Status']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print("Training model...")
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model
MODEL_PATH = "loan_approval_model.pkl"
joblib.dump(model, MODEL_PATH)
print(f"Model saved at: {os.path.abspath(MODEL_PATH)}")

# Verify the file exists after saving
if os.path.exists(MODEL_PATH):
    print("Model file verified on disk.")
else:
    raise FileNotFoundError(f"Model file was not created at: {os.path.abspath(MODEL_PATH)}")