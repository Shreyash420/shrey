import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
data = {
    "Outlook": ["Rainy", "Rainy", "Overcast", "Sunny", "Sunny", "Sunny", "Overcast", "Rainy", "Rainy", "Sunny", "Rainy", "Overcast", "Overcast", "Sunny"],
    "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool", "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"],
    "Humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "Normal", "High", "Normal", "High"],
    "Windy": [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
    "PlayGolf": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
}

df = pd.DataFrame(data)


categorical_columns = ["Outlook", "Temperature", "Humidity", "Windy"]
for column in categorical_columns:
    df[column] = df[column].astype('category').cat.codes

X = df.drop("PlayGolf", axis=1)
y = df["PlayGolf"].astype('category').cat.codes


clf = DecisionTreeClassifier(criterion="entropy", random_state=0)
clf.fit(X, y)


plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X.columns, class_names=["No", "Yes"], filled=True)
plt.show()


def calculate_entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


initial_entropy = calculate_entropy(y)


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

for feature in X.columns:
    ig = calculate_information_gain(df, feature, "PlayGolf")
    print(f"Information Gain for {feature}: {ig:.4f}")

predictions = clf.predict(X)
accuracy = accuracy_score(y, predictions)
print(f"Accuracy of the Decision Tree: {accuracy:.2f}")

