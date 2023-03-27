import matplotlib.pyplot as plt;
import numpy as np;

plt.subplots(figsize=(15,5));

sizelist = [3, 8, 100];

for i in range(3):
    size=sizelist[i];
    X, Y = np.meshgrid(np.linspace(0, 10, size+1),
                       np.linspace(0, 10, size+1))
    C = np.linspace(0, 100, size*size).reshape(size, size);
    plt.subplot(1, 3, i+1);
    plt.pcolormesh(X, Y, C, cmap="rainbow");
plt.show();