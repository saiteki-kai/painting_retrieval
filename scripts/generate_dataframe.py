from cv2 import normalize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

corrupted = ['69008.jpg', '121.jpg', '38324.jpg', '97976.jpg', '84772.jpg', '77094.jpg', '85232.jpg', '80945.jpg',
             '32150.jpg', '1262.jpg', '32577.jpg', '43658.jpg', '65430.jpg', '95897.jpg', '83271.jpg', '84021.jpg',
             '32192.jpg', '50789.jpg', '38922.jpg']

df = pd.read_csv("../data/raw/dataset/all_data_info.csv")
df.rename(columns={"new_filename": "filename"}, inplace=True)
df.drop(columns=["pixelsx", "pixelsy", "size_bytes", "artist_group", "source"], inplace=True)
df.dropna(subset=["genre"], inplace=True)
df.drop(df.loc[df['filename'].isin(corrupted)].index, inplace=True)
df.reset_index(drop=True, inplace=True)

# save memory
df["artist"] = df["artist"].astype("category")
df["style"] = df["style"].astype("category")
# df["date"] = pd.to_datetime(df["date"])

print(df.memory_usage(deep=True))
print(df.info())

train_genres = df.loc[df["in_train"]]['genre'].value_counts(normalize=True).rename('percent').reset_index()
fig = plt.figure(figsize=(15, 4))
ax = sns.barplot(x='index', y='percent', data=train_genres, order=df.loc[df["in_train"]]['genre'].value_counts().index)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
#fig.savefig("genres-all.png")
plt.show()

# ------------------------------------------------------------
df_train = df.loc[df["in_train"]]
c_train = Counter(df_train["genre"])

genres = []
for key, value in c_train.items():
    if (value / len(df_train) * 100) < 1:
        genres.append(key)

others = df['genre'].str.contains('|'.join(genres))
df.loc[others, 'genre'] = 'other'

# save memory
df["genre"] = df["genre"].astype("category")

print(df.memory_usage(deep=True))
print(df.info())

train_genres = df.loc[df["in_train"]]['genre'].value_counts(normalize=True).rename('percent').reset_index()
fig = plt.figure(figsize=(15, 4))
ax = sns.barplot(x='index', y='percent', data=train_genres, order=df.loc[df["in_train"]]['genre'].value_counts().index)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
#fig.savefig("genres-reducted.png")
plt.show()

# sample a subset for faster testing -------------------------

df_train = df.loc[df["in_train"]]
df_test = df.loc[~df["in_train"]]
N_train = len(df_train) # Just to set
N_test = len(df_test) # Just to set
subsample = False


print(f"Initial train size: {len(df_train)}")
print(f"Initial test size: {len(df_test)}")

N_train = 110 # If we want to resize
N_test = 0 # If we want to resize

if N_train < len(df_train):
    df_train = (
        df_train.groupby("genre", group_keys=False)
        .apply(lambda x: x.sample(int(np.rint(N_train * len(x) / len(df_train)))))
        .sample(frac=1)
        .reset_index(drop=True)
    )
    subsample = True

if N_test < len(df_test):
    df_test = (
        df_test.groupby("genre", group_keys=False)
        .apply(lambda x: x.sample(int(np.rint(N_test * len(x) / len(df_test)))))
        .sample(frac=1)
        .reset_index(drop=True)
    )
    subsample = True

print(f"Final train size: {len(df_train)}")
print(f"Final test size: {len(df_test)}")

df = pd.concat([df_train, df_test])
df.reset_index(drop=True, inplace=True)

# save dataset -------------------------

if subsample:
    print("Subsampled.")
    df.to_pickle("../data/raw/dataset/data_info_subsampled.pkl")
else:
    print("Original size.")
    df.to_pickle("../data/raw/dataset/data_info.pkl")
    # df.to_csv("../data/raw/dataset/data_info.csv") # 'data_info_subsampled.csv'


# from collections import Counter

# c_train = Counter(df_train["genre"])
# c_test = Counter(df_test["genre"])

# for key, value in c_train.items():
#     if (value / len(df_train) * 100) >= 1:
#         print(
#             f"{key:25s}\t{(value / len(df_train)*100):5.3f}% \t{(c_test[key]/len(df_test)*100):5.3f}% "
#         )

# s_train = 0
# s_test = 0
# for key, value in c_train.items():
#     if (value / len(df_train) * 100) < 1:
#         s_train = s_train + value
#         s_test = s_test + c_test[key]

# print(f"{s_train} {s_train / len(df_train) * 100}")
# print(f"{s_test} {s_test / len(df_test) * 100}")
