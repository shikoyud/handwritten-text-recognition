import pandas as pd

data_file_path = "data/raw/iam_words/words.txt"

labels = pd.read_csv(
        data_file_path,
        sep=" ",
        comment="#",
        header=None
        )

print(labels.head())
print(labels.shape)
