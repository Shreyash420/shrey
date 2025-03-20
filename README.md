import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Step 1: Define the dataset
data = {
    "Outlook": ["Rainy", "Rainy", "Overcast", "Sunny", "Sunny", "Sunny", "Overcast", "Rainy", "Rainy", "Sunny", "Rainy", "Overcast", "Overcast", "Sunny"],
    "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool", "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"],
    "Humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "Normal", "High", "Normal", "High"],
    "Windy": [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
    "PlayGolf": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
}

df = pd.DataFrame(data)

# Step 2: Convert categorical variables to numerical
categorical_columns = ["Outlook", "Temperature", "Humidity", "Windy"]
for column in categorical_columns:
    df[column] = df[column].astype('category').cat.codes

X = df.drop("PlayGolf", axis=1)
y = df["PlayGolf"].astype('category').cat.codes

# Step 3: Build the decision tree classifier
clf = DecisionTreeClassifier(criterion="entropy", random_state=0)
clf.fit(X, y)

# Step 4: Visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X.columns, class_names=["No", "Yes"], filled=True)
plt.show()

# Step 5: Calculate information gain (entropy)
def calculate_entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# Compute entropy for the whole dataset
initial_entropy = calculate_entropy(y)

# Compute entropy after splitting on each feature
def calculate_information_gain(df, feature, target):
    total_entropy = calculate_entropy(df[target])
    
    # Compute the weighted entropy of each subset
    values, counts = np.unique(df[feature], return_counts=True)
    weighted_entropy = np.sum([
        (counts[i] / np.sum(counts)) * calculate_entropy(df[df[feature] == values[i]][target])
        for i in range(len(values))
    ])
    
    # Information gain is the difference between total entropy and weighted entropy
    information_gain = total_entropy - weighted_entropy
    return information_gain

# Display information gain for each feature
for feature in X.columns:
    ig = calculate_information_gain(df, feature, "PlayGolf")
    print(f"Information Gain for {feature}: {ig:.4f}")

# Step 6: Evaluate the model
predictions = clf.predict(X)
accuracy = accuracy_score(y, predictions)
print(f"Accuracy of the Decision Tree: {accuracy:.2f}")
# shrey
coder
