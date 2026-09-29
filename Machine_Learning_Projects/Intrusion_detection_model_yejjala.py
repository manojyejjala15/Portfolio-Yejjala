# Import necessary libraries for data handling, machine learning, and evaluation
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# Load the CICIDS2017 dataset from a CSV file
# Replace the path with your actual file location
data = pd.read_csv("E:/workspace/CICIDS2017_sample.csv")

# --- Preprocessing ---
# Drop rows with missing values to ensure data integrity
data = data.dropna()
# Remove duplicate rows to avoid redundancy in training
data = data.drop_duplicates()

# Encode the target 'Label' column into numeric values (e.g., Benign=0, Attack1=1, etc.)
label_encoder = LabelEncoder()
data["Label"] = label_encoder.fit_transform(data["Label"])
# Calculate the number of unique classes for reference
num_classes = len(np.unique(data["Label"]))

# Keep only numeric columns, as machine learning models require numerical input
data = data.select_dtypes(include=[np.number])

# Optional: Outlier removal using IQR method (commented out)
# Note: In intrusion detection, outliers might represent attacks, so removal is skipped here
"""
Q1 = data.quantile(0.25)  # First quartile
Q3 = data.quantile(0.75)  # Third quartile
IQR = Q3 - Q1  # Interquartile range
threshold = 1.5  # Standard threshold for outlier detection
lower_bound = Q1 - threshold * IQR  # Lower limit for acceptable values
upper_bound = Q3 + threshold * IQR  # Upper limit for acceptable values
# Remove rows where any feature is outside the bounds
data = data[~((data < lower_bound) | (data > upper_bound)).any(axis=1)]
"""

# --- Feature and Target Preparation ---
# Separate features (X) and target (y)
X = data.drop("Label", axis=1)  # Features are all columns except 'Label'
y = data["Label"]  # Target is the 'Label' column

# Normalize features to have zero mean and unit variance
# This helps models like SVM and Logistic Regression converge faster
scaler = StandardScaler()
X = scaler.fit_transform(X)

# --- Data Splitting ---
# Split data into training (80%) and testing (20%) sets
# random_state ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Classifier Definitions ---
# Define a dictionary of classifiers with tuned hyperparameters
classifiers = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,  # Number of trees
        max_depth=10,  # Limit depth to prevent overfitting
        random_state=42,  # For reproducibility
    ),
    "SVM": SVC(
        kernel="rbf",  # Radial basis function kernel for non-linear separation
        C=10,  # Higher penalty for misclassification to improve accuracy
        gamma="scale",  # Adaptive gamma based on data (1 / (n_features * X.var()))
        random_state=42,
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,  # Increase iterations for convergence
        multi_class="multinomial",  # Explicitly handle multiple classes
        solver="lbfgs",  # Suitable solver for multinomial logistic regression
        random_state=42,
    ),
    "Gradient Boost": GradientBoostingClassifier(
        max_depth=3,  # Limit depth of individual trees
        n_estimators=100,  # Number of boosting stages
        learning_rate=0.1,  # Step size for updates (default value)
        random_state=42,
    ),
}

# --- Training and Evaluation ---
# Initialize an empty list to store results
results = []

# Iterate through each classifier
for clf_name, clf in classifiers.items():
    # Train the classifier on the training data
    clf.fit(X_train, y_train)
    # Predict on training and testing sets
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    # Calculate training and testing errors as 1 - accuracy
    training_error = 1 - accuracy_score(y_train, y_train_pred)
    testing_error = 1 - accuracy_score(y_test, y_test_pred)

    # Evaluate fit based on error difference with a threshold
    error_evaluation = "Good Fit"
    error_diff = training_error - testing_error
    threshold = 0.05  # Threshold to determine significant over/underfitting
    if error_diff < -threshold:  # Training error much lower than testing
        error_evaluation = "Overfitting"
    elif error_diff > threshold:  # Training error much higher than testing
        error_evaluation = "Underfitting"

    # Append results to the list
    results.append([clf_name, training_error, testing_error, error_evaluation])

    # Print detailed classification report for each classifier
    # Includes precision, recall, and F1-score per class
    print(f"\nClassification Report for {clf_name}:")
    print(
        classification_report(
            y_test, y_test_pred, target_names=label_encoder.classes_
        )
    )

# --- Display Results ---
# Convert results list to a DataFrame for tabular display
result_df = pd.DataFrame(
    results,
    columns=[
        "Classifier",
        "Training Error",
        "Testing Error",
        "Error Evaluation",
    ],
)
print("\nResults Table:")
print(result_df)
