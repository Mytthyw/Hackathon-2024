import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score  

# Load datasets
df = pd.read_csv("Surveydata_train_(2).csv")
dftest = pd.read_csv("Surveydata_test_(2).csv")
dftravel = pd.read_csv('Traveldata_train_(2).csv')
dftraveltest = pd.read_csv('Traveldata_test_(2).csv')

# Combine survey and travel data on 'ID'
train_data = pd.merge(df, dftravel, on='ID', how='inner')
test_data = pd.merge(dftest, dftraveltest, on='ID', how='inner')

# Check the info of the combined training data
train_data.info()

# Prepare features and target for training
X_train = train_data.drop(columns=["Overall_Experience"])
y_train = train_data["Overall_Experience"]

# Prepare the test set
test_ID = test_data["ID"]
X_test = test_data.drop(columns=["ID"]) 

# Combine train and test for dummy variable creation
combined_data = pd.concat([X_train, X_test], keys=["train", "test"])

# Convert categorical variables to dummy variables
combined_data = pd.get_dummies(combined_data)

# Split back into train and test sets
X_train_encoded = combined_data.loc["train"]
X_test_encoded = combined_data.loc["test"]

# Ensure the columns match by reindexing
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# Initialize and train the decision tree model
model = DecisionTreeClassifier()  # Or DecisionTreeRegressor for regression
model.fit(X_train_encoded, y_train)

# Make predictions on the encoded test set
test_predictions = model.predict(X_test_encoded)

# Save predictions to CSV with ID and predicted outcome
output_df = pd.DataFrame({"ID": test_ID, "PredictedOutcome": test_predictions})
output_df.to_csv("predictions.csv", index=False)
