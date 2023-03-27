from sklearn.datasets import make_moons;
import matplotlib.pyplot as plt;
import pandas as pd;

x, y = make_moons(
    random_state=3,
    noise=0,
    n_samples=300
)

df = pd.DataFrame(x);
df["target"]=y;

df0=df[df["target"]==0];
df1=df[df["target"]==1];

plt.figure(figsize=(5,5));
plt.scatter(df0[0], df0[1], color="b", alpha=0.5);
plt.scatter(df1[0], df1[1], color="r", alpha=0.5);
plt.grid();
plt.show();
