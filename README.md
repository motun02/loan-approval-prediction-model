# Loan Approval Prediction

A beginner machine learning project: predicting whether a loan application is Approved or Denied from applicant details, built as a learning project during an internship in a bank's analytics department.

**Data:** a public loan dataset from Kaggle (`loan.csv`, 61 applicants). No data, screenshots or information from any employer was used.

**Tools:** Python, Pandas, Scikit-learn, Jupyter notebook in VS Code.

**Headline result:** both models scored 100% on one train/test split and about 97% under cross-validation. A single income threshold (about 53,500) almost completely separates the two groups, so the data is easy. This does not show the model would work on real bank data.

## Files

- `loan.ipynb`: the full notebook (inspection, EDA, models, evaluation)
- `loan.csv`: the dataset
- `README.md`: this report

## 1. The problem

A bank must decide whether to approve or deny each loan application. Using past applications where the answer is known, I trained a model to learn the pattern and predict the answer for new applications.

This is a **classification** problem, because the answer is a category. If the answer were a number (a loan amount or an interest rate), it would be regression.

Terms used throughout:

- **Features:** the clues about each applicant (age, income, credit score), called `X` in the code.
- **Target:** the answer to predict (`loan_status`), called `y` in the code.
- **Row:** one applicant. The model makes one prediction per row.

## 2. The data

61 applicants (rows) and 8 columns.

| Column | Type | Meaning |
|---|---|---|
| age | number | Applicant age |
| gender | text | 2 values |
| occupation | text | 38 different values |
| education_level | text | 5 values |
| marital_status | text | 2 values |
| income | number | Applicant income |
| credit_score | number | Credit score |
| loan_status | text | Approved or Denied (the target) |

### Inspecting the data

**Why:** before building anything I needed the size of the data, the type of each column and whether anything was missing.

```python
import pandas as pd
df = pd.read_csv("loan.csv")
df.head()        # first 5 rows
df.info()        # rows, columns, types, missing values
df.describe()    # summary of the numeric columns
```

**Found:** 61 rows, 8 columns, no missing values. Three numeric columns and five text columns.

## 3. Exploring the data

**Why:** exploring shows what patterns exist and what a model might learn.

```python
df["loan_status"].value_counts()
df["loan_status"].value_counts(normalize=True)
df.groupby("loan_status")[["age", "income", "credit_score"]].median()
df.groupby("loan_status")[["age", "income", "credit_score"]].agg(["min", "max"])
```

**Findings:**

1. 45 applicants were Approved and 16 Denied, so 74% were approved (45 / 61).
2. **Baseline accuracy is 74%.** A "model" that says Approved for everyone would be right 74% of the time. Any real model must beat this.
3. Approved applicants have higher income and credit scores. Mean income is about 92,956 (Approved) vs 39,688 (Denied). Median credit score is 750 vs 615.
4. The groups barely overlap. The highest Denied income is 52,000 and the lowest Approved income is 55,000. Credit scores overlap only between 650 and 660, with three applicants in that range.
5. Age shows a pattern but is not a rule. All 8 applicants under 28 were Denied, but ages 28 to 33 appear in both groups. Age may simply track income.

These are observations about this dataset, not statements about how any bank actually decides.

## 4. Splitting into training and test sets

**Why:** if a model is trained and tested on the same rows, it can memorize them and look perfect. So some rows are hidden for testing.

```python
from sklearn.model_selection import train_test_split

X = df[["age", "income", "credit_score"]]   # features
y = df["loan_status"]                        # target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
```

- `test_size=0.25` hides a quarter of the rows: **45 for training, 16 for testing**.
- `random_state=42` makes the shuffle repeatable.
- `stratify=y` keeps the Approved/Denied mix the same in both parts. Only 16 applicants are Denied, so without it a random split can leave the test set with very few of them. Across six different seeds without stratify, the Denied count in the test set ranged from 2 to 7. With it, the test set is always 12 Approved and 4 Denied.

## 5. Model 1: Logistic regression

**Why:** a simple, standard first model for yes/no predictions. It learns how strongly each feature pushes an applicant toward Approved or Denied and combines them into a probability.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)   # 1. create
model.fit(X_train, y_train)                 # 2. learn from training data
predictions = model.predict(X_test)         # 3. predict on hidden test data
```

**Result:** all 16 test applicants were predicted correctly (100%), compared with the 74% baseline.

## 6. Evaluating the model

**Why:** accuracy alone hides which mistakes were made, and for a bank the two kinds of mistake cost different amounts.

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
```

- **Accuracy:** correct predictions divided by total (16 / 16 = 1.0).
- **Confusion matrix:** a table of right and wrong calls. Rows are real answers, columns are predictions, in the order Approved, Denied. Mine was `[[12, 0], [0, 4]]`: no mistakes of either kind.
- **Precision:** when the model says Approved, how often is it right?
- **Recall:** of everyone who really was Approved, how many did the model find?
- **F1:** one score that balances precision and recall.
- **Support:** how many real cases of each class are in the test set.

**Which mistake is worse for a bank?** Wrongly approving someone who should be denied is usually costlier, because that loan may default and become bad debt. Wrongly denying a good applicant loses a customer and the interest income, which is usually the smaller cost. A bank would therefore watch precision for Approved and recall for Denied.

### An age-only experiment

To see what errors look like, I trained a model on age alone. Its confusion matrix was `[[11, 1], [0, 4]]`: one good applicant wrongly denied, and no one wrongly approved. Precision for Denied was 4/5 (0.80) and recall for Approved was 11/12 (0.92), which shows why precision and recall say more than accuracy.

## 7. Text columns and a fairness decision

**Why:** models can only do maths on numbers, so text columns must be converted. This is also a fairness question: treating applicants differently by gender or marital status is a legal and ethical problem in lending.

One-hot encoding turns each text value into its own yes/no column.

```python
def score(features):
    X = pd.get_dummies(df[features], drop_first=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    m = LogisticRegression(max_iter=5000).fit(X_tr, y_tr)
    return accuracy_score(y_te, m.predict(X_te))
```

`drop_first=True` removes one redundant column per feature.

**Occupation was excluded:** it has 38 different values among only 61 applicants, so encoding it would create about 37 columns with one or two people each. The model could memorize individuals instead of learning patterns (overfitting).

| Features | Accuracy |
|---|---|
| Numbers only | 1.0 |
| Numbers + education | 1.0 |
| Numbers + education + gender + marital status | 1.0 |
| Education only | 0.8125 |

**Conclusions:**

- The text columns added nothing, because the numbers alone already scored 100%. There is no reason to include gender and marital status in a credit model: they add risk and no benefit.
- Education alone scored 81.25% (13 of 16). Approving everyone gets 12 of 16 on this test set, so education beat the baseline by one applicant, a very weak signal that also shows how much one row moves the score (6.25 points).

## 8. Model 2: Decision tree

**Why:** a decision tree learns a flowchart of yes/no questions that a person can read. It showed how the answer was reached, and a second method agreeing with the first suggests the data is driving the result.

```python
from sklearn.tree import DecisionTreeClassifier, export_text

tree = DecisionTreeClassifier(max_depth=2, random_state=42)
tree.fit(X_train, y_train)
print(export_text(tree, feature_names=list(X_train.columns)))
```

`max_depth=2` keeps the tree small and readable and stops it memorizing every applicant.

**Result:** the tree needed one question: is income 53,500 or less? If yes, Denied; if no, Approved. It scored 100% and did not use age or credit score.

53,500 sits in the gap between the highest Denied income (52,000) and the lowest Approved income (55,000). This is the real reason for the perfect scores: the dataset can be split by a single income threshold. A filter in Excel would do the same job.

## 9. Cross-validation

**Why:** one split gives one score, and it depends on which 16 applicants landed in the test set. Cross-validation tests several times to get a steadier number: the data is split into five parts, the model trains on four and is tested on one, rotating until each part has been the test set once, and the scores are averaged.

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5)
print(scores)
print(scores.mean())
```

**Result:** scores of 1.0, 0.917, 0.917, 1.0 and 1.0, averaging about **0.967 (97%)**.

The single split said 100%, but the steadier estimate is about 97%. The first split was lucky and contained no hard cases. Cross-validation gives every applicant a turn in the test set, including those near the income and credit score boundary, where an occasional mistake appears (each 0.917 is one mistake out of 12). 97% is still strong and well above the 74% baseline, but it is a more honest number than 100%.

## 10. Results summary

| Step | Finding |
|---|---|
| Data | 61 applicants, 8 columns, no missing values, 74% approved |
| Baseline | 74% (approving everyone) |
| Logistic regression | 100% on one split |
| Decision tree | 100%, using one income split at about 53,500 |
| Cross-validation | About 97% average |
| Fairness | Gender and marital status excluded: no benefit, added risk |
| Occupation | Excluded: 38 values in 61 rows |

## 11. Limitations

- **Small dataset:** 61 rows, with 16 in the test set. One applicant changes the score by 6.25 points.
- **Data that is too easy:** income separates the groups almost completely. Real lending data overlaps much more, with missing values, errors and far more rows. This model could not be used at a real bank as it is.
- **Possibly synthetic data:** the clean separation suggests the data may be artificial or heavily tidied, but I cannot confirm that.
- **Observation, not policy:** the income threshold describes this dataset. It is not how any bank actually decides.
- **Fairness and compliance:** a real credit model would need fairness checks and approval through a bank's own risk and compliance processes.

## 12. Next steps

- Use a larger, messier loan dataset so the groups overlap and the model faces hard decisions.
- Use cross-validation as the standard way to report scores.
- Study precision and recall properly once the model makes real errors.
- Build a Power BI dashboard of the findings.
- Practise the same questions in SQL by loading the data into SQL Server.

## 13. How to run

1. Install Python, then install the Python and Jupyter extensions (both by Microsoft) in VS Code.
2. Run: `pip install pandas matplotlib scikit-learn ipykernel`
3. Keep `loan.csv` in the same folder as `loan.ipynb`.
4. Select your Python as the notebook kernel, then use Run All.

## Glossary

- **Baseline:** the score of a trivial approach (here, approve everyone). A real model must beat it.
- **Classification:** predicting a category. **Regression:** predicting a number.
- **Feature:** an input clue. **Target:** the answer being predicted.
- **Training set:** rows the model learns from. **Test set:** hidden rows used to judge it.
- **Stratify:** keep the same class mix in the training and test sets.
- **Overfitting:** memorizing the training data instead of learning general patterns.
- **One-hot encoding:** turning a text column into yes/no number columns.
- **Confusion matrix:** a table of correct and wrong predictions by class.
- **Precision, recall, F1:** how reliable the model's calls are, how many true cases it finds, and a balance of the two.
- **Cross-validation:** testing on several different splits and averaging the scores.
