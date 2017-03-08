from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.util import MLUtils
import shutil

if __name__ == '__main__':
	"""pre-processing data"""
	iris = open("iris.data.txt", "r").readlines()
	output_string = ""
	for row in iris:
		line = row.split(",")
		if "Iris-setosa" in line[-1]:
			new_string = "1 "
		elif "Iris-virginica" in line[-1]:
			new_string = "3 "
		elif "Iris-versicolor" in line[-1]:
			new_string = "3 "
		# new_string = line[0] + " "
		count = 1
		for i in line[:-1]:
			new_string += str(count) + ":" + str(i) + " "
			count += 1
		new_string += "\n"
		output_string += new_string
		# print(repr(row))
		# print(repr(new_string))
		# break
	output_file = open("iris.txt", "w")
	output_file.write(output_string)
	output_file.close()

	sc = SparkContext()
	sc.setLogLevel('ERROR')
	data = MLUtils.loadLibSVMFile(sc, "iris.txt")
	training, test = data.randomSplit([0.8, 0.2])
	model = NaiveBayes.train(training, 1.0)
	# Make prediction and test accuracy.
	predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
	accuracy = 1.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / test.count()
	print('model accuracy {}'.format(accuracy))
