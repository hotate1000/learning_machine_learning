from sklearn.datasets import make_blobs;
import pandas as pd;
import matplotlib.pyplot as plt;
from sklearn.model_selection import train_test_split;
from sklearn import svm;
from sklearn.metrics import accuracy_score;
import numpy as np;
from matplotlib.colors import ListedColormap;

X, y = make_blobs(
    random_state=0,
    n_features=2,
    centers=2,
    cluster_std=1,
    n_samples=300)

df = pd.DataFrame(X);
df["target"] = y;
df.head();
df0 = df[df["target"]==0];
df1 = df[df["target"]==1];
plt.figure(figsize=(5, 5));
plt.scatter(df0[0], df0[1], color="b", alpha=0.5);
plt.scatter(df1[0], df1[1], color="r", alpha=0.5);
# plt.show();

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0);

df = pd.DataFrame(X_train);
df["target"] = y_train;
df0 = df[df["target"]==0];
df1 = df[df["target"]==1];
plt.figure(figsize=(5, 5));
plt.scatter(df0[0], df0[1], color="b", alpha=0.5);
plt.scatter(df1[0], df1[1], color="r", alpha=0.5);
plt.title("train:75%");
# plt.show();

df = pd.DataFrame(X_test);
df["target"] = y_test;
df0 = df[df["target"]==0];
df1 = df[df["target"]==1];
plt.figure(figsize=(5, 5));
plt.scatter(df0[0], df0[1], color="b", alpha=0.5);
plt.scatter(df1[0], df1[1], color="r", alpha=0.5);
plt.title("test:25%");
# plt.show();

model = svm.SVC();
model.fit(X_test, y_test);

pred = model.predict(X_test);
df = pd.DataFrame(X_test);
df["target"]=pred;
df0=df[df["target"]==0];
df1=df[df["target"]==1];
plt.figure(figsize=(5,5));
plt.scatter(df0[0], df0[1], color="b", alpha=0.5);
plt.scatter(df1[0], df1[1], color="r", alpha=0.5);
plt.title("predict");
# plt.show();

pred = model.predict(X_test);
score = accuracy_score(y_test, pred);
print("正解率：", score*100, "%");

pred = model.predict([[1,3]]);
print("1,3=", pred);

pred = model.predict(([[1,2]]));
print("1,2=", pred);

plt.figure(figsize=(5,5));
plt.scatter(df0[0], df0[1], color="b", alpha=0.5);
plt.scatter(df1[0], df1[1], color="r", alpha=0.5);
plt.scatter([1], [3], color="b", marker="x", s=300);
plt.scatter([1], [2], color="r", marker="x", s=300);
plt.title("predict");
# plt.show();


def plot_boundary(model, X, Y, target, xlabel, ylabel):
    cmap_dots = ListedColormap(["#1f77b4", "#ff7f0e", "#2ca02c"]);
    cmap_files = ListedColormap(["#c6dcec", "#ffdec2", "#cae7ca"]);

    plt.figure(figsize=(5,5));
    if model:
        XX, YY = np.meshgrid(
            np.linspace(X.min()-1, X.max()+1, 200),
            np.linspace(Y.min()-1, Y.max()+1, 200));
        pred = model.predict(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape);
        plt.pcolormesh(XX, YY, pred, cmap=cmap_files,shading="auto");
        plt.contour(XX, YY, pred, colors="gray");
    plt.scatter(X, Y, c=target, cmap=cmap_dots);
    plt.xlabel(xlabel);
    plt.ylabel(ylabel);
    plt.show();


df=pd.DataFrame(X_test);
pred = model.predict(X_test);

plot_boundary(model, df[0], df[1], pred, "df[0]", "df[1]");