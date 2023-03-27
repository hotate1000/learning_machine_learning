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
print("正解率：", score*100, "%");

plt.figure(figsize=(5,5));
plt.scatter(x, y, color="b", alpha=0.5);
plt.plot(x, model.predict(x), color='red');
plt.grid();
plt.show();
