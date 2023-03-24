import pandas as pd;
import matplotlib.pyplot as plt
from sklearn import datasets;


iris = datasets.load_iris();

df = pd.DataFrame(iris.data);
# print(df.head());
# print("--------------------------------");

df.columns = iris.feature_names;
df["target"] = iris.target;
print(df.head());
print("--------------------------------");

df0 = df[df["target"]==0];
df1 = df[df["target"]==1];
df2 = df[df["target"]==2];

plt.figure(figsize=(5, 5));
xx1 = "sepal width (cm)";
df0[xx1].hist(color="b", alpha=0.5);
df1[xx1].hist(color="r", alpha=0.5);
df2[xx1].hist(color="g", alpha=0.5);
plt.xlabel(xx1);
plt.show();
print("--------------------------------");

xx2 = "sepal width (cm)";
yy2 = "sepal length (cm)";
plt.figure(figsize=(5,5));
plt.scatter(df0[xx2], df0[yy2], color="b", alpha=0.5);
plt.scatter(df1[xx2], df1[yy2], color="r", alpha=0.5);
plt.scatter(df2[xx2], df2[yy2], color="g", alpha=0.5);
plt.xlabel(xx2);
plt.ylabel(yy2);
plt.grid();
plt.show();
