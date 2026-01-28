from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import *
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


# Indlæs datasættet
iris = load_iris()


# Konverter til en pandas DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df['target_name'] = iris_df['target'].apply(lambda x: iris.target_names[x])

# Vis de første rækker i datasættet
#print(iris_df.head())


# Scatterplot for to funktioner
#plt.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target, cmap='viridis')
#plt.xlabel(iris.feature_names[0])
#plt.ylabel(iris.feature_names[1])
#plt.title('Scatterplot of Iris Data')
#plt.show()


# Opdel data
# Stratificeret opdeling
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=67, stratify=iris.target
)

print(f"Antal træningsdata: {len(X_train)}")
print(f"Antal testdata: {len(X_test)}")


# Træn modellen
#model = LogisticRegression(max_iter=200)
#model = KNeighborsClassifier(n_neighbors=20)
model = DecisionTreeClassifier(max_depth=4, random_state=67)
model.fit(X_train, y_train)

# Forudsig med testdata
y_pred = model.predict(X_test)

#print("Forudsigelser:", y_pred)

# Evaluer nøjagtighed
accuracy = accuracy_score(y_test, y_pred)
print(f"Modelens nøjagtighed: {accuracy:.2f}")

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=iris.target_names, cmap='viridis')
plt.title("Confusion Matrix")
plt.show()


plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Klassifikationstræ")
plt.show()