from sklearn.datasets import make_regression;
from sklearn.metrics import accuracy_score;
import pandas as pd;
import matplotlib.pyplot as plt;


x, y = make_regression(
    random_state=3,
    n_features=1,
    noise=20,
    n_samples=30
);
# print(x);
# print(y);

df = pd.DataFrame(x);
# plt.figure(figsize=(5,5));
# plt.scatter(df[0], y, color="b", alpha=0.5);
# plt.grid();
# plt.show();


from sklearn.linear_model import LinearRegression;
from sklearn.metrics import r2_score;
from sklearn.model_selection import train_test_split;


X_train1, X_test1, Y_train1, Y_test1 = train_test_split(x, y, random_state=0);
# print(X_train1);
# print("-----");
# print(X_test1);
# print("-----");
# print(Y_train1);
# print("-----");
# print(Y_test1);
# print("-----");

model = LinearRegression();
model.fit(X_train1, Y_train1);

pred = model.predict(X_test1);
score = r2_score(Y_test1, pred);
# print("正解率：", score*100, "%");

# plt.figure(figsize=(5,5));
# plt.scatter(x, y, color="b", alpha=0.5);
# plt.plot(x, model.predict(x), color="red");
# plt.grid();
# plt.show();


x, y = make_regression(
    random_state=3,
    n_samples=30,
    n_features=1,
    noise=80
);

X_train2, X_test2, Y_train2, Y_test2 = train_test_split(x, y, random_state=0);

model = LinearRegression();
model.fit(X_train2, Y_train2);

pred = model.predict(X_test2);
score = r2_score(Y_test2, pred);
# print("正解率：", score*100, "%");

# plt.figure(figsize=(5,5));
# plt.scatter(x, y, color="b", alpha=0.5);
# plt.plot(x, model.predict(x), color='red');
# plt.grid();
# plt.show();


import numpy as np;
import matplotlib.pyplot as plt;
from matplotlib.colors import ListedColormap;

def plot_boundary(model, x, y, target, xlabel, ylabel):
    cmap_dots = ListedColormap(["#1f77b4", "#ff7f0e", "#cae7ca"]);
    cmap_fills = ListedColormap(["#c6dcec", "#ffdec2", "#cae7ca"]);
    # plt.figure(figsize=(5, 5));
    if model:
        XX, YY = np.meshgrid(
            np.linspace(x.min()-1, x.max()+1, 200),
            np.linspace(y.min()-1, y.max()+1, 200)
        )
        pred = model.predict(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape);
        plt.pcolormesh(XX, YY, pred, cmap=cmap_fills, shading="auto");
        plt.contour(XX, YY, pred, cmap="gray");
        plt.scatter(x, y, c=target, cmap=cmap_dots);
        plt.xlabel(xlabel);
        plt.ylabel(ylabel);
        plt.show();

from sklearn.datasets import make_blobs;

x, y = make_blobs(
    random_state=0,
    n_features=2,
    centers=3,
    cluster_std=1,
    n_samples=300
)

df = pd.DataFrame(x);
# print(df.head());
# print(y);
plot_boundary(None, df[0], df[1], y, "df[0]", "df[1]");

from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LogisticRegression;
from sklearn.metrics import accuracy_score;

X_train, X_test, Y_train, Y_test = train_test_split(x, y, random_state=0);

model = LogisticRegression();
model.fit(X_train, Y_train);

pred = model.predict(X_test);
score = accuracy_score(Y_test, pred);
print("正解率:", score*100, "%");

df = pd.DataFrame(X_test);
plot_boundary(model, df[0], df[1], Y_test, "df[0]", "df[1]");
