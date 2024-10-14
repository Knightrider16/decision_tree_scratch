import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('data.csv')


def calculate_entropy(y):
    # Calculate the entropy of the target variable
    value_counts = y.value_counts()
    total = len(y)
    entropy = 0

    for count in value_counts:
        probability = count / total
        entropy -= probability * np.log2(probability)

    return entropy


def calculate_information_gain(data, feature, target):
    # Calculate Information Gain for a given feature
    total_entropy = calculate_entropy(data[target])
    value_counts = data[feature].value_counts()
    total = len(data)
    weighted_entropy = 0

    for value, count in value_counts.items():
        subset = data[data[feature] == value]
        weighted_entropy += (count / total) * calculate_entropy(subset[target])

    return total_entropy - weighted_entropy


def find_best_feature(data, target):
    # Find the best feature to split on
    features = data.columns[:-1]  # All columns except the target
    best_gain = 0
    best_feature = None

    for feature in features:
        gain = calculate_information_gain(data, feature, target)
        if gain > best_gain:
            best_gain = gain
            best_feature = feature

    return best_feature


def create_decision_tree(data, target):
    # Base case: if all target values are the same, return that value
    if len(data[target].unique()) == 1:
        return data[target].iloc[0]

    # Base case: if no features left, return the most common target value
    if len(data.columns) == 1:
        return data[target].mode()[0]

    # Find the best feature to split on
    best_feature = find_best_feature(data, target)
    tree = {best_feature: {}}

    # Split the data on the best feature
    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        tree[best_feature][value] = create_decision_tree(subset, target)

    return tree


# Create the decision tree
decision_tree = create_decision_tree(df, 'buys_computer')


# Display the decision tree
def print_tree(tree, depth=0):
    for key, value in tree.items():
        print("  " * depth + str(key))
        if isinstance(value, dict):
            print_tree(value, depth + 1)
        else:
            print("  " * (depth + 1) + str(value))


print_tree(decision_tree)


def predict(tree, sample):
    # Base case: if the tree is a leaf node (i.e., not a dictionary)
    if not isinstance(tree, dict):
        return tree
    # Get the root feature of the current tree
    root_feature = next(iter(tree))

    # Get the value of the root feature from the sample
    feature_value = sample[root_feature]

    # Check if the feature value exists in the tree
    if feature_value in tree[root_feature]:
        # Recur down to the corresponding subtree
        subtree = tree[root_feature][feature_value]
        return predict(subtree, sample)
    else:
        # Handle the case where the feature value does not exist in the tree
        return None  # or return a default value or the most common class

tree = create_decision_tree(df, 'buys_computer')

for i in range(len(df)):
    data = df.iloc[i]
    prediction = predict(tree, data)

    print(f"actual prediction: {data.loc['buys_computer']}")
    print(f"prediction: buys computer? {prediction}")

    print()

