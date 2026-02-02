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
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.tree import plot_tree
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("kundanbedmutha/exam-score-prediction-dataset")

print("Path to dataset files:", path)
print(os.listdir(path))

# Indlæs datasættet
dataSet = pd.read_csv(os.path.join(path, "Exam_Score_Prediction.csv"))
print(dataSet.head())

X = dataSet.drop(columns=["exam_score", "student_id"])
Y = dataSet["exam_score"]

categorical_features = ["gender", "course", "internet_access", "sleep_quality", "study_method", "exam_difficulty", "facility_rating"]
numeric_features = [col for col in X.columns if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(), categorical_features)
    ]
)

model = Pipeline([
    ("preprocess", preprocessor),
    ("regressor", DecisionTreeRegressor(max_depth=4, random_state=67))
])
# Konverter til en pandas DataFrame
#DataSet_df = pd.DataFrame(dataSet=Data.dataSet, columns=Data.feature_names)
#DataSet_df['target'] = Data.target 
#DataSet_df['target_name'] = DataSet_df['target'].apply(lambda x: Data.target_names[x])

# Vis de første rækker i datasættet
#print(DataSet_df.head())

#Split

# Scatterplot for to funktioner
#plt.scatter(Data.dataSet[:, 0], Data.dataSet[:, 1], c=Data.target, cmap='viridis')
#plt.xlabel(Data.feature_names[0])
#plt.ylabel(Data.feature_names[1])  
#plt.title('Scatterplot of Iris Data')
#plt.show()


# Opdel dataSet
# Stratificeret opdeling
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=67
)

print(f"Antal træningsdata: {len(X_train)}")
print(f"Antal testdata: {len(X_test)}")


# Træn modellen
#model = LogisticRegression(max_iter=200)
#model = KNeighborsClassifier(n_neighbors=20)
#model = DecisionTreeClassifier(max_depth=4, random_state=67)
model.fit(X_train, y_train)

# Forudsig med testdata
y_pred = model.predict(X_test)


#print("Forudsigelser:", y_pred)

# Evaluer nøjagtighed
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.2f}")
#accuracy = accuracy_score(y_test, y_pred)
#print(f"Modelens nøjagtighed: {accuracy:.2f}")

#ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=Data.target_names, cmap='viridis')
#plt.title("Confusion Matrix")
#plt.show()

tree = model.named_steps["regressor"]

plt.figure(figsize=(16, 8))
plot_tree(
    tree,
    feature_names=model.named_steps["preprocess"].get_feature_names_out(),
    filled=True,
    rounded=True
)
plt.title("Decision Tree Regressor – Exam Score Prediction")
plt.show()
#plt.figure(figsize=(12, 8))
#plot_tree(model, feature_names=Data.feature_names, class_names=Data.target_names, filled=True)
#plt.title("Klassifikationstræ")
#plt.show()