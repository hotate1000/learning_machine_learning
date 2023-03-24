from sklearn import datasets;

iris = datasets.load_iris();

print(iris);
print("--------------------------------");

print("学習用のデータ\n", iris.data);
print("特徴量の名前\n", iris.feature_names);
print("目的の値\n", iris.target);
print("目的の名前\n", iris.target_names);
print("データーセットの説明\n", iris.DESCR);
print("--------------------------------");
