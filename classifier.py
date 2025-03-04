import pandas as pd
from ydata_profiling import ProfileReport

data = pd.read_csv("data/train.csv")
x_test = pd.read_csv("data/train.csv")

# profile = ProfileReport(data, title="Disaster Tweets", explorative=True)
# profile.to_file("tweets.html")

print(data.info())
print(data["keyword"].value_counts())
print(data["location"].value_counts())
