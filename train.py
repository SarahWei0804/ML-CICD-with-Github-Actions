import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import skops.io as sio

df = pd.read_csv('data/heart.csv')
X = df.drop("HeartDisease", axis=1)
y = df['HeartDisease'].values
train_x, test_x, train_y, test_y = train_test_split(X, y, stratify=y, random_state=0, test_size=0.3)

cat_cols = [list(X.columns).index(i) for i in list(X.select_dtypes('object').columns)]
num_cols = list(set([i for i in range(len(X.columns))]) - set(cat_cols))

transformer = ColumnTransformer([
    ('encoder', OrdinalEncoder(), cat_cols),
    ('num_imputer', SimpleImputer(strategy='median'), num_cols),
    ('num_scaler', StandardScaler(), num_cols)
])
pipe = Pipeline(steps=[
    ('preprocessing', transformer),
    ('model', RandomForestClassifier(n_estimators=300, random_state=0))
])
pipe.fit(train_x.values, train_y)


predictions = pipe.predict(test_x.values)
accuracy = accuracy_score(test_y, predictions)
f1 = f1_score(test_y, predictions, average='macro')

with open("result/metrics.txt", "w") as outfile:
    outfile.write(f"\nAccuracy = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}.")

cm = confusion_matrix(test_y, predictions, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot()
plt.savefig("result/model_results.png", dpi=120)

sio.dump(pipe, "model/heart_pipeline.skops")