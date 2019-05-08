# Import Library
from sklearn import svm


# Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]


# Create SVM(Support Vector Machine) classification object 
model = svm.svc(gamma='scale') 


# Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)


# Predict Output
predicted= model.predict(x_test)

