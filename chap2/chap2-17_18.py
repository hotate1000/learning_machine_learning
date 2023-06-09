from sklearn.datasets import make_circles;
from sklearn.datasets import make_gaussian_quantiles;
import pandas as pd;
import matplotlib.pyplot as plt;

# x, y = make_circles(
#     random_state=3,
#     noise=0.1,
#     n_samples=300
# )

x,y = make_gaussian_quantiles(
    random_state=3,
    n_features=2,
    n_classes=3,
    n_samples=300
)

df=pd.DataFrame(x);
df["target"]=y;
df0=df[df["target"]==0];
df1=df[df["target"]==1];
df2=df[df["target"]==2];

plt.figure(figsize=(5,5));
plt.scatter(df0[0],df0[1],color="b",alpha=0.5);
plt.scatter(df1[0],df1[1],color="r",alpha=0.5);
plt.scatter(df2[0],df2[1],color="g",alpha=0.5);
plt.grid();
plt.show();