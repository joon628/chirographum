from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
from sklearn.metrics import accuracy_score

# read data --- (*1)
digits = datasets.load_digits()
x = digits.images
y = digits.target
x = x.reshape((-1, 64)) # Convert 2d matrix into 1d

# divide data into test and train
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# train data
clf = svm.LinearSVC()
clf.fit(x_train, y_train)

# print expectation, print output accuracy
y_pred = clf.predict(x_test)
print(accuracy_score(y_test, y_pred))
