import nltk
nltk.data.path.append('./nltk_data')
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from nltk.stem import WordNetLemmatizer

data = pd.read_csv("data/train.csv")
x_test = pd.read_csv("data/test.csv")

lemmatizer = WordNetLemmatizer()

def clean_text(text):
  text = text.lower()
  text = re.sub(r"http\S+|www\S+", "", text)
  text = re.sub(r'@\w+', '', text)
  text = re.sub(r"[^a-zA-Z\s]", "", text)
  text = text.strip()
  text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
  return text

data["text"] = data["text"].apply(clean_text)
x_test["text"] = x_test["text"].apply(clean_text)

print(data)

# plt.figure(figsize=(6, 4))
# sns.histplot(data[data["target"] == 0]["location"], color="blue", label="Not Disaster", kde=True)
# sns.histplot(data[data["target"] == 1]["location"], color="red", label="Disaster", kde=True)
# plt.legend()
# plt.show()

target = "target"

x = data.drop(target, axis=1)
y = data[target]
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)

nom_transformer = Pipeline(steps=[
  ("imputer", SimpleImputer(strategy="most_frequent")),
  ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
  ("nom_feature", nom_transformer, ["keyword", "location"]),
  # ("text_feature", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=0.01, max_df=0.95), "text"),
  ("text_feature", TfidfVectorizer(stop_words="english", ngram_range=(1, 2)), "text"),
])

cls = Pipeline(steps=[
  ("preprocessor", preprocessor),
  ("model", RandomForestClassifier()),
  # ("model", SVC()),
])

# result = cls.fit_transform(x_train, y_train)
# feature_names = preprocessor.get_feature_names_out()
# selector = cls.named_steps["feature_selector"]
# selected_features = feature_names[selector.get_support()]
# print(selected_features)
# print("len:", len(selected_features))
# print(pd.DataFrame(result.todense()))

randomForestParams = {
  "model__n_estimators": [100, 200, 300],
  "model__criterion": ["gini", "entropy", "log_loss"],
  "model__max_depth": [None, 2],
}

# svcParams = {
#   "model__C": [1.0, 2.0],
#   "model__kernel": ["linear", "poly", "rbf"],
# }

grid_search = GridSearchCV(estimator=cls, param_grid=randomForestParams, cv=4, scoring="accuracy", verbose=2, n_jobs=-1)
grid_search.fit(x_train, y_train)
y_valid_predicted = grid_search.predict(x_valid)

print(classification_report(y_valid, y_valid_predicted))
print(grid_search.best_score_)
print(grid_search.best_params_)

y_test_predicted = grid_search.predict(x_test)

submission = pd.DataFrame({
  "id": x_test["id"],
  "target": y_test_predicted
})
submission.to_csv("data/sample_submission.csv", index=False)
