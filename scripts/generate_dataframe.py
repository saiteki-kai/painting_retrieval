import pandas as pd
import numpy as np

df = pd.read_csv("./data/raw/dataset/all_data_info.csv")
df.rename(columns={"new_filename": "filename"}, inplace=True)
df.drop(
    columns=["pixelsx", "pixelsy", "size_bytes", "artist_group", "source"], inplace=True
)
df.dropna(subset=["genre"], inplace=True)
df.reset_index(drop=True, inplace=True)

# save memory
df["artist"] = df["artist"].astype("category")
df["genre"] = df["genre"].astype("category")
df["style"] = df["style"].astype("category")
# df["date"] = pd.to_datetime(df["date"])

print(df.memory_usage(deep=True))
print(df.info())

# sample a subset for faster testing -------------------------

N_train = 1000
N_test = 500

df_train = df.loc[df["in_train"]]
df_test = df.loc[~df["in_train"]]

df_train = (
    df_train.groupby("genre", group_keys=False)
    .apply(lambda x: x.sample(int(np.rint(N_train * len(x) / len(df_train)))))
    .sample(frac=1)
    .reset_index(drop=True)
)
df_test = (
    df_test.groupby("genre", group_keys=False)
    .apply(lambda x: x.sample(int(np.rint(N_test * len(x) / len(df_test)))))
    .sample(frac=1)
    .reset_index(drop=True)
)

df = pd.concat([df_train, df_test])
df.reset_index(drop=True, inplace=True)
# ------------------------------------------------------------

df.to_pickle("./data/raw/dataset/data_info.pkl")
# df.to_csv("./data/raw/dataset/data_info.csv")

# fig = plt.figure(figsize=(15, 4))
# ax = sns.countplot(df.loc[df["in_train"]]["genre"])
# ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
# fig.savefig("genres1.png")
# plt.show()


from collections import Counter

c_train = Counter(df_train["genre"])
c_test = Counter(df_test["genre"])

for key, value in c_train.items():
    if (value / len(df_train) * 100) >= 1:
        print(
            f"{key:25s}\t{(value / len(df_train)*100):5.3f}% \t{(c_test[key]/len(df_test)*100):5.3f}% "
        )

s_train = 0
s_test = 0
for key, value in c_train.items():
    if (value / len(df_train) * 100) < 1:
        s_train = s_train + value
        s_test = s_test + c_test[key]

print(f"{s_train} {s_train / len(df_train) * 100}")
print(f"{s_test} {s_test / len(df_test) * 100}")
