# hello

print("-------------------------------------")

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

# scripy
import scipy

print("scipy: {}".format(scipy.__version__))

# numpy
import numpy

print("numpy: {}".format(numpy.__version__))

# matplotlib
import matplotlib

print("matplotlib: {}".format(matplotlib.__version__))

# pandas
import pandas

print("pandas: {}".format(pandas.__version__))

# scikit-learn
import sklearn

print("sklearn: {}".format(sklearn.__version__))


# load libraries

from pandas import read_csv  # <-- resposible for reading data
from pandas.plotting import scatter_matrix  # <--
from matplotlib import pyplot as plt  # <--
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pickle

# load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
collums = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
dataset = read_csv(url, names=collums)

print("-------------------------------------")

# shape
print(dataset.shape)

print("-------------------------------------")

# head
print(dataset.head(20))

print("-------------------------------------")

# descriptions
print(dataset.describe())
print("-------------------------------------")

# class distribution
print(dataset.groupby("class").size())

# box and whisker plots
dataset.plot(kind="box", subplots=True, layout=(2, 2), sharex=False, sharey=False)
plt.show()

# histograms
dataset.hist()
plt.show()

# scatter plot matrix
scatter_matrix(dataset)
plt.show()

# Split-Out validation dataset
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]

X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, Y, test_size=0.20, random_state=1
)

filename = "Finalized_model_test.sav"

if os.path.exists(filename):
    print("-------------------------------------\n")
    print(f"File Found. loading {filename}.")
    print("\n-------------------------------------")
    with open(filename, "rb") as f:
        loaded_model = pickle.load(f)
    # load the model from disk
    resultFromPickle = loaded_model.score(X_validation, Y_validation)
    predictions = loaded_model.predict(X_validation)
    print(resultFromPickle)
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
else:

    print("-------------------------------------\n")
    print("File Not Found.")
    print("\n-------------------------------------")

    # Spot Check Algorithims

    models = []

    models.append(("LR", LogisticRegression(max_iter=1000)))
    models.append(("LDA", LinearDiscriminantAnalysis()))
    models.append(("KNN", KNeighborsClassifier()))
    models.append(("CART", DecisionTreeClassifier()))
    models.append(("NB", GaussianNB()))
    models.append(("SVM", SVC(gamma="auto")))

    # Evaluate each model in run

    results = []
    names = []
    topResult = 0.0
    topResultName = ""

    for name, model in models:
        kFold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_result = cross_val_score(
            model, X_train, Y_train, cv=kFold, scoring="accuracy"
        )
        if cv_result.mean() > topResult:
            topResult = cv_result.mean()
            topResultName = name
        results.append(cv_result)
        names.append(name)
        print("%s: %f (%f)" % (name, cv_result.mean(), cv_result.std()))
    print("-------------------------------------")
    print()
    print(f"{name} is the top model with {topResult.mean():.2f}% accuracy!!")
    print()
    print("-------------------------------------")

    # Compare Algorithims

    plt.boxplot(results, tick_labels=names)
    plt.title("Algorithim Comparison")
    plt.show()

    # Make predictions on validation dataset

    model = SVC(gamma="auto")
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)

    # save the model to disk
    # filename = "Finalized_model.sav"
    pickle.dump(model, open(filename, "wb"))

    # Evaluate predictions

    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))


# help(“FunctionName”) help syntax in Python to learn about all of the functions that you’re using.

print("Done!")
