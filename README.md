# Iris Flower Classification — My First Machine Learning Project in Python

A beginner machine learning project in Python that classifies iris flowers into three species based on their measurements. This project follows the [Your First Machine Learning Project in Python Step-By-Step](https://machinelearningmastery.com/machine-learning-in-python-step-by-step/) guide by Jason Brownlee at MachineLearningMastery.com.

## About the Project

The goal is to walk through a complete, end-to-end machine learning workflow on a small, well-known dataset:

1. Load the dataset
2. Summarize the data
3. Visualize the data
4. Evaluate several algorithms
5. Make predictions and check accuracy

## Dataset

This project uses the classic **Iris flower dataset**, which contains 150 samples split evenly across three species:

- *Iris-setosa*
- *Iris-versicolor*
- *Iris-virginica*

Each sample has four numeric features:

| Feature | Unit |
|---|---|
| Sepal length | cm |
| Sepal width | cm |
| Petal length | cm |
| Petal width | cm |

## Technologies Used

- Python 3
- [scikit-learn](https://scikit-learn.org/) — models and evaluation
- [pandas](https://pandas.pydata.org/) — loading and summarizing data
- [NumPy](https://numpy.org/) — numerical operations
- [Matplotlib](https://matplotlib.org/) — plots
- [SciPy](https://scipy.org/) — scientific computing

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. Install the dependencies:
   ```bash
   pip install numpy scipy matplotlib pandas scikit-learn
   ```

## Usage

Run the main script:

```bash
python main.py
```

The script will:

- Print the dataset's shape, first rows, statistical summary, and class distribution
- Show box-and-whisker plots, histograms, and a scatter plot matrix
- Compare six algorithms using 10-fold stratified cross-validation
- Train the best model and report its accuracy on a held-out validation set

## Algorithms Evaluated

| Abbreviation | Algorithm |
|---|---|
| LR | Logistic Regression |
| LDA | Linear Discriminant Analysis |
| KNN | K-Nearest Neighbors |
| CART | Classification and Regression Trees (Decision Tree) |
| NB | Gaussian Naive Bayes |
| SVM | Support Vector Machine |

The data is split 80/20 into training and validation sets. Each model is scored with stratified 10-fold cross-validation on the training set, and the best performer is then used to make predictions on the validation set, reported with an accuracy score, confusion matrix, and classification report.

## Results

In the guide, **Support Vector Machine (SVM)** scored the highest accuracy, at about 98% during cross-validation and around 96% on the validation set. Your exact numbers may vary slightly.

<!-- Add your own results here, e.g.:
| Model | Mean Accuracy | Std Dev |
|---|---|---|
| LR | 0.00 | 0.00 |
-->

## What I Learned

- How to load and explore a dataset with pandas
- How to visualize data to spot patterns between features
- How to compare multiple models fairly with cross-validation
- How to evaluate a final model with a confusion matrix and classification report

## Acknowledgments

- Tutorial: [Your First Machine Learning Project in Python Step-By-Step](https://machinelearningmastery.com/machine-learning-in-python-step-by-step/) by Jason Brownlee, MachineLearningMastery.com
- Dataset: R.A. Fisher's Iris dataset, via the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/53/iris)
