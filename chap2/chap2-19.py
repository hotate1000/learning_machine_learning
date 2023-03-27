from sklearn.datasets import make_regression;
import pandas as pd;
import matplotlib.pyplot as plt;

x, y=make_regression(
    random_state=3,
    n_features=1,
    noise=10,
    bias=100,
    n_samples=300
)

df = pd.DataFrame(x);
plt.figure(figsize=(5,5));
plt.scatter(df[0],y,color="b",alpha=0.5);
plt.grid();
plt.show();
