# hello
import time
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

# ---------- DATA ----------
from pandas import read_csv                                    # <-- loads a CSV file (local or from a URL) into a table called a DataFrame

# ---------- CHARTS ----------
from pandas.plotting import scatter_matrix                     # <-- draws a grid of scatter plots comparing every column against every other column
from matplotlib import pyplot as plt                           # <-- the main charting tool; plt.show() opens the chart window, plt.title() adds a title, etc.

# ---------- SPLITTING & TESTING ----------
from sklearn.model_selection import train_test_split           # <-- splits data into a training set and a test set (e.g. 80% / 20%)
from sklearn.model_selection import cross_val_score            # <-- trains and tests a model several times on different slices of the data, returns the scores
from sklearn.model_selection import StratifiedKFold            # <-- decides HOW cross_val_score slices the data (keeps each species evenly mixed in every slice)

# ---------- GRADING RESULTS ----------
from sklearn.metrics import classification_report               # <-- per-class report card: precision, recall, f1-score
from sklearn.metrics import confusion_matrix                    # <-- table of "what it guessed" vs "what the answer was"
from sklearn.metrics import accuracy_score                      # <-- a single number: fraction of guesses that were right (0.0 to 1.0)

# ---------- THE MODELS (different ways to learn) ----------
from sklearn.linear_model import LogisticRegression             # <-- draws straight-line boundaries between classes; simple, fast, good first try
from sklearn.tree import DecisionTreeClassifier                 # <-- learns a flowchart of yes/no questions ("is petal length > 2.5?")
from sklearn.neighbors import KNeighborsClassifier              # <-- looks at the K closest known examples and goes with the majority
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # <-- uses statistics (averages/spread of each class) to find the best dividing lines
from sklearn.naive_bayes import GaussianNB                      # <-- uses probability: "given these measurements, which class is most likely?"
from sklearn.svm import SVC                                     # <-- finds the boundary with the widest possible gap between classes

# ---------- SAVING ----------
import pickle                                    # <-- saves any Python object (like a trained model) to a file, and loads it back later   

# load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
collums = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
dataset = read_csv(url, names=collums)

if os.path.exists("iris.csv"):
    dataset = read_csv("iris.csv")              # load the local copy
else:
    dataset = read_csv(url, names=collums)      # download it
    dataset.to_csv("iris.csv", index=False)     # save it for next time

print("-------------------------------------")

# shape
print(dataset.shape)

print("-------------------------------------")

# head
print(dataset.head(100))

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
    
    # load the model from disk
    
    print("-------------------------------------\n")
    print(f"File Found. loading {filename}.")
    print("\n-------------------------------------")
    with open(filename, "rb") as f:
        model = pickle.load(f)
else:
    # if no file loaded the model get trained and makes a file
    
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
        print(f"{name}: {cv_result.mean()*100:.2f}% ({cv_result.std()*100:.2f}%)")
    print("-------------------------------------")
    print()
    print(f"{topResultName} is the top model with {topResult.mean():.2f}% accuracy!!")
    print()
    print("-------------------------------------")

    # Compare Algorithims

    plt.boxplot(results, tick_labels=names)
    plt.title("Algorithim Comparison")
    plt.show()

    # Make predictions on validation dataset

    model = dict(models)[topResultName]
    model.fit(X_train, Y_train)

    # save the model to disk
    # filename = "Finalized_model.sav"
    with open(filename, "wb") as f:
        pickle.dump(model, f)

    # Evaluate predictions


    

predictions = model.predict(X_validation)
accuracyScore = accuracy_score(Y_validation, predictions)
print(f"Accuracy Score: {accuracyScore:.2f}%")
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

new_flower = [[5.7, 2.8, 4.1, 1.3]]   # sepal length, sepal width, petal length, petal width
print(model.predict(new_flower))

# help(“FunctionName”) help syntax in Python to learn about all of the functions that you’re using.
print("-------------------------------------")

# checks how long it took to run not that it really matters

timeRan = time.process_time()
print(f"\nran in {timeRan:.2f} Seconds")
print("\nDone!\n")
print("-------------------------------------")
