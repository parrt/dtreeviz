import pandas as pd
import numpy as np

import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import Imputer
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, DecisionTreeClassificationModel
from pyspark.ml.regression import DecisionTreeRegressor, DecisionTreeRegressionModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType,StructField, StringType, IntegerType, DoubleType


import dtreeviz

import os
import sys

random_state = 1234 # get reproducible trees
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

spark = SparkSession.builder \
            .master("local[2]") \
            .appName("dtreeviz_sparkml") \
            .getOrCreate()


dataset_url = "https://raw.githubusercontent.com/parrt/dtreeviz/master/data/titanic/titanic.csv"
dataset = pd.read_csv(dataset_url)

df_schema = StructType([StructField("PassengerId",IntegerType(),True),
                StructField("Survived",IntegerType(),True),
                StructField("Pclass",IntegerType(),True),
                StructField("Name",StringType(),True),
                StructField("Sex",StringType(),True),
                StructField("Age",DoubleType(),True),
                StructField("SibSp",IntegerType(),True),
                StructField("Parch",IntegerType(),True),
                StructField("Ticket",StringType(),True),
                StructField("Fare",DoubleType(),True),
                StructField("Cabin",StringType(),True),
                StructField("Embarked",StringType(),True)])

data = spark.createDataFrame(dataset, df_schema)

sex_label_indexer = StringIndexer(inputCol="Sex", outputCol="Sex_label", handleInvalid="keep")
embarked_label_indexer = StringIndexer(inputCol="Embarked", outputCol="Embarked_label", handleInvalid="keep")
age_imputer = Imputer(inputCols=["Age"], outputCols=["Age"])

features = ["Pclass", "Sex_label", "Embarked_label", "Age", "SibSp", "Parch", "Fare"]
target = "Survived"

vector_assembler = VectorAssembler(inputCols=features, outputCol="features")
decision_tree = DecisionTreeClassifier(featuresCol="features", labelCol="Survived", maxDepth=4, seed=1234)
pipeline = Pipeline(stages=[sex_label_indexer, embarked_label_indexer, age_imputer, vector_assembler, decision_tree])
model = pipeline.fit(data)

# extract from the pipeline the tree model classifier
tree_model_classifier = model.stages[4]

# recompute the dataset on which the model was trained
dataset = Pipeline(stages=[sex_label_indexer, embarked_label_indexer, age_imputer]) \
    .fit(data) \
    .transform(data) \
    .toPandas()[features + [target]]


viz_model = dtreeviz.model(tree_model_classifier,
                           X_train=dataset[features], y_train=dataset[target],
                           feature_names=features,
                           target_name=target, class_names=["survive", "perish"])


x = dataset[features].iloc[10]

viz_model.view(x=x).show()

