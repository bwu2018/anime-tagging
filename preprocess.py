import pandas as pd
import os
import shutil
import requests

root_dir = "./"

df = pd.read_csv("all_data.csv")
df = df.head(1000)
df = df[df["rating"] == "s"]
df = df.drop(columns = ["rating", "created_at", "score", "preview_url"])
df = df.set_index(["id", "sample_url", "sample_width", "sample_height"]).apply(lambda x: x.str.split().explode()).reset_index()
top_100 = list(df["tags"].value_counts()[:100].index)
top_df = df[df["tags"].isin(top_100)]
dataset_path = root_dir + 'anime-dataset'
shutil.rmtree(dataset_path)
os.mkdir(dataset_path)
for tag in top_df["tags"].value_counts().keys()[:100]:
  os.mkdir(dataset_path + '/' + tag)

tags = []

for i, row in top_df.iterrows():
  # print(row)
  r = requests.get("https:" + row['sample_url'])
  if r.status_code == 404:
      continue
  f = open(dataset_path + '/' + row['tags'] + '/' + str(row['id']) + '.txt', 'w')
  tags.append(row['tags'])
  f.write(row['sample_url'])
  f.close()

print("Total:", len(set(tags)))