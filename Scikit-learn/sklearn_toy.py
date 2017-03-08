from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
# load the iris datasets
dataset = datasets.load_iris()
# fit a Naive Bayes model to the data
model = MultinomialNB()
model.fit(dataset.data, dataset.target)
# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))