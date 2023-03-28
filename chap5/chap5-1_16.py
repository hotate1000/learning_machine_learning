import pandas as pd;
from sklearn.datasets import load_digits;
import matplotlib.pyplot as plt;
from sklearn.model_selection import train_test_split;
from sklearn import svm;
from sklearn.metrics import accuracy_score;
from PIL import Image;
from pathlib import Path;
import sys;
sys.path.append(str(Path(__file__).resolve().parent.parent));
import numpy as np;


digits = load_digits();
# print(digits);

df = pd.DataFrame(digits.data);
# print(df);

# for i in range(10):
#     plt.subplot(1, 10, i+1);
#     plt.axis("off");
#     plt.title(digits.target[i]);
#     # plt.imshow(digits.data[i:i+1], cmap="Greys");
#     plt.imshow(digits.images[i], cmap="Greys");
# plt.show();

X = digits.data;
y = digits.target;

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0);
# print("train=", len(X_train));
# print("test=", len(X_test));

model = svm.SVC(kernel="rbf", gamma=0.001);
model.fit(X_train, y_train);
# print(model);

pred = model.predict(X_test);
score = accuracy_score(y_test, pred);
# print("正解率：", score*100, "%");

image = Image.open("sample_code/4.png").convert("L");
plt.imshow(image, cmap="gray");
image = image.resize((8,8), Image.ANTIALIAS);
plt.imshow(image, cmap="gray");
# plt.show();


img = np.asarray(image, dtype=float);
# print(img);

img = 16 - np.floor(17*img/256);
# print(img);

img = img.flatten();
# print(img);
# print(digits.data[0:1]);

predict = model.predict([img]);
print("予想=", predict);

image = Image.open("sample_code/6.png").convert("L");
image = image.resize((8,8), Image.ANTIALIAS);
img = np.asarray(image, dtype=float);
img = 16 - np.floor(17*img/256);
img = img.flatten();

predict = model.predict([img]);
print("予想=", predict);

plt.imshow(image, cmap="gray");
plt.show();
