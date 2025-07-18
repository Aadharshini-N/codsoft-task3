# codsoft-task3
This project uses the Iris flower dataset, a classic dataset in machine learning, to train a model that classifies flowers into three species: Setosa, Versicolor, and Virginica. The classification is based on four features: sepal length, sepal width, petal length, and petal width.
# importing libraries
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# Load dataset from sklearn
iris = datasets.load_iris()
X = iris.data  # features (sepal length, sepal width, petal length, petal width)
y = iris.target  # target (species)

# Convert to DataFrame for better visualization
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
print(df.head())
import pandas as pd
from sklearn.datasets import load_iris
# Load dataset
iris = load_iris()
# Convert to DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df["species"] = [iris.target_names[i] for i in iris.target]
# Check for nulls
print("Checking for missing values:")
print(iris_df.isnull().sum())
# Drop rows with nulls (if any)
iris_df_cleaned = iris_df.dropna()
# Confirm cleanup
print("\n Dataset cleaned. Any remaining nulls?")
print(iris_df_cleaned.isnull().sum())
# Optional: Show shape before and after cleaning
print(f"\nOriginal shape: {iris_df.shape}")
print(f"Cleaned shape:  {iris_df_cleaned.shape}")


# 1. Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train a classifier (e.g., KNN)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_scaled, y_train)

# 5. Predict
y_pred = model.predict(X_test_scaled)

# 6. Plot the Confusion Matrix Heatmap
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues",
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
# showing distribution based on species
for feature in features:
    plt.figure(figsize=(7, 4))
    for species in iris.target_names:
        subset = iris_df_fixed[iris_df_fixed["species"] == species]
        plt.hist(subset[feature], bins=10, alpha=0.6, label=species)
    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.legend()
    plt.show()
# showing distribution based on petal length
plt.figure(figsize=(6, 5))
sns.scatterplot(data=iris_df_fixed, x="petal length (cm)", y="petal width (cm)", hue="species", palette="Set2")
plt.title("Petal Length vs Width by Species")
plt.show()
for feature in features:
    plt.figure(figsize=(6, 4))
    sns.violinplot(data=iris_df_fixed, x="species", y=feature, hue="species", palette="Pastel1", legend=False)

    plt.title(f"{feature.capitalize()} Distribution by Species")
    plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming iris_df_fixed and features are already defined
# If not, load and prepare as shown earlier

for feature in features:
    plt.figure(figsize=(6, 4))
    sns.histplot(
        data=iris_df_fixed,
        x=feature,
        hue="species",
        multiple="stack",  # can also try "dodge" or "layer"
        palette="Set2",
        edgecolor="black"
    )
    plt.title(f"{feature.capitalize()} Histogram by Species")
    plt.xlabel(feature.capitalize())
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
