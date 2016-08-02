from __future__ import division
import re
import operator
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold


# Check Survival.ipynb for steps

# LOAD TRAINING AND TEST DATA  --------------------------------

training_df = pd.read_csv("train.csv")
testing_df = pd.read_csv("test.csv")

# CLEAN UP / FIX DATA  --------------------------------

training_df["Age"] = training_df["Age"].fillna(training_df["Age"].median())
training_df["Fare"] = training_df["Fare"].fillna(training_df["Fare"].median())
training_df["Embarked"] = training_df["Embarked"].fillna("S")

training_df.loc[training_df["Sex"] == "male", "Sex"] = 0
training_df.loc[training_df["Sex"] == "female", "Sex"] = 1
training_df.loc[training_df["Embarked"] == "S", "Embarked"] = 0
training_df.loc[training_df["Embarked"] == "C", "Embarked"] = 1
training_df.loc[training_df["Embarked"] == "Q", "Embarked"] = 2

testing_df["Age"] = testing_df["Age"].fillna(testing_df["Age"].median())
testing_df["Fare"] = testing_df["Fare"].fillna(testing_df["Fare"].median())
testing_df["Embarked"] = testing_df["Embarked"].fillna("S")

testing_df.loc[testing_df["Sex"] == "male", "Sex"] = 0
testing_df.loc[testing_df["Sex"] == "female", "Sex"] = 1
testing_df.loc[testing_df["Embarked"] == "S", "Embarked"] = 0
testing_df.loc[testing_df["Embarked"] == "C", "Embarked"] = 1
testing_df.loc[testing_df["Embarked"] == "Q", "Embarked"] = 2

# FEATURE FUNCTIONS / VARIABLES  --------------------------------

training_df["FamilySize"] = training_df["SibSp"] + training_df["Parch"]
training_df["NameLength"] = training_df["Name"].apply(lambda x: len(x))

testing_df["FamilySize"] = testing_df["SibSp"] + testing_df["Parch"]
testing_df["NameLength"] = testing_df["Name"].apply(lambda x: len(x))

title_mapping = {
    "Mr": 1,
    "Miss": 2,
    "Mrs": 3,
    "Master": 4,
    "Dr": 5,
    "Rev": 6,
    "Major": 7,
    "Col": 7,
    "Mlle": 8,
    "Mme": 8,
    "Don": 9,
    "Lady": 10,
    "Countess": 10,
    "Jonkheer": 10,
    "Sir": 9,
    "Capt": 7,
    "Ms": 2,
    "Dona": 10
}

# extract title from passengers name
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

family_id_mapping = {}

# extract last name and assign / add family id 
def get_family_id(row):
    # find last name 
    last_name = row["Name"].split(",")[0]
    # create family id
    family_id = "{0}{1}".format(last_name, row["FamilySize"])
    # look up the id in the mapping
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            # Get the max id from the mapping and add one if
            # we don't have an id
            current_id = (max(family_id_mapping.items(), 
                key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]


# APPLY FEATURES   --------------------------------

training_titles = training_df["Name"].apply(get_title)
testing_titles = testing_df["Name"].apply(get_title)

for k, v in title_mapping.items():
    training_titles[training_titles == k] = v
    testing_titles[testing_titles == k] = v

training_df["Title"] = training_titles
testing_df["Title"] = testing_titles


# Get the family ids with the apply method
family_ids = training_df.apply(get_family_id, axis=1)
test_family_ids = testing_df.apply(get_family_id, axis=1)

# Categorize familysize < 3 into one
family_ids[training_df["FamilySize"] < 3] = -1
test_family_ids[testing_df["FamilySize"] < 3] = -1

training_df["FamilyId"] = family_ids
testing_df["FamilyId"] = test_family_ids


# ASSIGN FEATURES   --------------------------------

predictors1 = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
               "FamilySize", "FamilyId", "Title"]

# selector = SelectKBest(f_classif, k=5)
# selector.fit(train[predictors], train["Survived"])

# scores = -np.log10(selector.pvalues_)

predictors2 = ["Pclass", "Embarked", "Sex", "Fare", "Title"]

# clf = RandomForestClassifier(random_state=1, n_estimators=150, 
#   min_samples_split = 8, min_samples_leaf=4)

#scores = cross_validation.cross_val_score(clf, train[predictors], 
#       train["Survived"], cv=3)

#print "random forest accuracy = ", scores.mean()

# DataQuest Ensemble algorithm recommendation

algorithms = [
    [RandomForestClassifier(random_state=1, n_estimators=200, 
  min_samples_split = 15, min_samples_leaf=8), predictors2],
    [GradientBoostingClassifier(random_state=1, n_estimators=25,
        max_depth=3), predictors2]
]

full_predictions = []
for alg, predictors in algorithms:
    # train the algorithm
    alg.fit(training_df[predictors], training_df["Survived"])

    # apply algorithm to test data
    predictions = alg.predict_proba(testing_df[predictors].astype(float))[:,1]

    # test accuracy on training data
    #predictions = alg.predict_proba(training_df[predictors].astype(float))[:,1]

    full_predictions.append(predictions)

# the gradient boosting classifier generates better predictions, 
#   so we weight it higher
predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4
predictions[predictions <= 0.5] = 0
predictions[predictions > 0.5] = 1
predictions = predictions.astype(int)

# accuracy = sum(predictions[predictions == training_df["Survived"]]) / len(predictions)
# print accuracy

#format for kaggle submission

submission = pd.DataFrame({
    "PassengerId": testing_df["PassengerId"],
    "Survived": predictions
    })

submission.to_csv("kaggle-titanic-2016-07-18.csv", index=False)
