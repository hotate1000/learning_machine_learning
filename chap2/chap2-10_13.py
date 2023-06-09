from sklearn.datasets import make_blobs;
import pandas as pd;
import matplotlib.pyplot as plt;

x, y = make_blobs(
    random_state=3,
    n_features=2,
    centers=5,
    cluster_std=1,
    n_samples=300
)

df = pd.DataFrame(x);
df["target"] = y;
print(df.head());
print("-----");

# df内でtargetが○の値を代入している
df0 = df[df["target"]==0];
df1 = df[df["target"]==1];
df2 = df[df["target"]==2];
df3 = df[df["target"]==3];
df4 = df[df["target"]==4];

plt.figure(figsize=(5,5));
plt.scatter(df0[0], df0[1], color="b", alpha=0.5);
plt.scatter(df1[0], df1[1], color="r", alpha=0.5);
plt.scatter(df2[0], df2[1], color="g", alpha=0.5);
plt.scatter(df3[0], df3[1], color="m", alpha=0.5);
plt.scatter(df4[0], df4[1], color="c", alpha=0.5);
plt.grid();
plt.show();
