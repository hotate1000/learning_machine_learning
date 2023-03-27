from sklearn.datasets import make_blobs;

x, y = make_blobs(
    random_state=0,
    n_features=2,
    centers=2,
    cluster_std=1,
    n_samples=300
)
# print(x);
# print(y);


import pandas as pd;

df = pd.DataFrame(x);
df["target"] = y;
df.head();
# print(df.head());


import matplotlib.pyplot as plt;

# df0 = df[df["target"]==0];
# df1 = df[df["target"]==1];
# plt.figure(figsize=(5,5));
# plt.scatter(df0[0], df0[1], color="b", alpha=0.5);
# plt.scatter(df1[0], df1[1], color="r", alpha=0.5);
# plt.show();


from sklearn.model_selection import train_test_split;

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=0);

df = pd.DataFrame(x_train);
df["target"]=y_train;
df0=df[df["target"]==0];
df1=df[df["target"]==1];
plt.figure(figsize=(5,5));
plt.scatter(df0[0], df0[1], color="b", alpha=0.5);
plt.scatter(df1[0], df1[1], color="r", alpha=0.5);
plt.title("train:75%");
plt.show();

df = pd.DataFrame(x_test);
df["target"]=y_test;
df0=df[df["target"]==0];
df1=df[df["target"]==1];
plt.figure(figsize=(5,5));
plt.scatter(df0[0], df0[1], color="b", alpha=0.5);
plt.scatter(df1[0], df1[1], color="r", alpha=0.5);
plt.title("train:25%");
plt.show();

