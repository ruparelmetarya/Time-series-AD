import tempfile
import urllib
import pandas as pd
import tensorflow as tf

#tutorial link: https://www.tensorflow.org/tutorials/wide

"""Downloading data """
train_file = tempfile.NamedTemporaryFile()
test_file = tempfile.NamedTemporaryFile()
urllib.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)
urllib.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.test", test_file.name)



"""Reading files into pandas dataframes"""
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]

# readcsv returns a data frame
df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)


LABEL_COLUMN = "label"
df_train[LABEL_COLUMN] = (df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(
    int)  # .apply takes in a condition- sets 1 if true, 0 o/w

df_test[LABEL_COLUMN] = (df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)



"""GROUPING COLUMNS: CATEGORICAL AND CONTINUOUS"""
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]


"""CONVERTING DATA INTO TENSORS"""

"""the input data is specified by means of an Input Builder function. 
This builder function will not be called until it is later passed to TF.Learn methods such as fit and evaluate. 
The purpose of this function is to construct the input data, which is represented in the form of tf.Tensors or 
tf.SparseTensors. 

In more detail, the Input Builder function returns the following as a pair:

feature_cols: A dict(ionary) from feature column names to Tensors or SparseTensors.
label: A Tensor containing the label column."""


def input_fn(df):
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values) #tf.constant creates a tensor: https://www.tensorflow.org/api_docs/python/tf/constant
                       for k in CONTINUOUS_COLUMNS}

    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {k: tf.SparseTensor( #returns sparsetensor
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        dense_shape=[df[k].size, 1])
        for k in CATEGORICAL_COLUMNS}
    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols.items() + categorical_cols.items()) #returns list of tuple pairs
    # Converts the label column into a constant Tensor.
   # label = tf.constant(df[LABEL_COLUMN].values)
    label = None
    # Returns the feature columns and the label.
    return feature_cols, label

def train_input_fn():
    return input_fn(df_train)

def eval_input_fn():
    return input_fn(df_test)



#NOTES:
"""Selecting and crafting the right set of feature columns is key to learning an effective model. 
A feature column can be either one of the raw columns in the original dataframe (let's call them base feature columns), 
or any new columns created based on some transformations defined over one or multiple base columns (let's call them 
derived feature columns). 
Basically, "feature column" is an abstract concept of any raw or derived variable that can be used to predict the target
 label."""
"""To define a feature column for a categorical feature,
 we can create a SparseColumn using the TF.Learn API. 
 If you know the set of all possible feature values of a column and there are only a few of them, you can use 
 sparse_column_with_keys. 
 Each key in the list will get assigned an auto-incremental ID starting from 0. 
 For example, for the gender column we can assign the feature string "Female" to an integer ID of 0 and "Male" 
 to 1 by doing """

gender = tf.contrib.layers.sparse_column_with_keys(
  column_name="gender", keys=["Female", "Male"])

"""HASH_BUCKETS FOR CATEGORICAL COLUMNS"""
"""What if we don't know the set of possible values in advance? Not a problem. We can use sparse_column_with_hash_bucket 
instead:"""

education = tf.contrib.layers.sparse_column_with_hash_bucket("education", hash_bucket_size=1000)

race = tf.contrib.layers.sparse_column_with_hash_bucket("race", hash_bucket_size=100)

marital_status = tf.contrib.layers.sparse_column_with_hash_bucket("marital_status", hash_bucket_size=100)

relationship = tf.contrib.layers.sparse_column_with_hash_bucket("relationship", hash_bucket_size=100)

workclass = tf.contrib.layers.sparse_column_with_hash_bucket("workclass", hash_bucket_size=100)

occupation = tf.contrib.layers.sparse_column_with_hash_bucket("occupation", hash_bucket_size=1000)

native_country = tf.contrib.layers.sparse_column_with_hash_bucket("native_country", hash_bucket_size=1000)


"""REAL-VALUED COLUMNS FOR CONTINUOUS COLUMNS"""
age = tf.contrib.layers.real_valued_column("age")
education_num = tf.contrib.layers.real_valued_column("education_num")
capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")


print 'AGE'
print age


"""NOTES- BUCKETIZATION"""
""" Sometimes the relationship between a continuous feature and the label is not linear. 
As an hypothetical example, a person's income may grow with age in the early stage of one's career, 
then the growth may slow at some point, and finally the income decreases after retirement.
 In this scenario, using the raw age as a real-valued feature column might not be a good choice because the model can 
 only learn one of the three cases:
    Income always increases at some rate as age grows (positive correlation),
    Income always decreases at some rate as age grows (negative correlation), or
    Income stays the same no matter at what age (no correlation)
If we want to learn the fine-grained correlation between income and each age group separately, we can leverage 
bucketization. 
Bucketization is a process of dividing the entire range of a continuous feature into a set of consecutive bins/buckets, 
and then converting the original numerical feature into a bucket ID (as a categorical feature) depending on which bucket
 that value falls into. 
So, we can define a bucketized_column over age as - """

age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])


"""CROSSED COLUMNS"""
education_x_occupation = tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4))

age_buckets_x_education_x_occupation = tf.contrib.layers.crossed_column(
  [age_buckets, education, occupation], hash_bucket_size=int(1e6))


model_dir = tempfile.mkdtemp()

feature_columns=[age]

print feature_columns
m = tf.contrib.learn.LinearClassifier(feature_columns,
  model_dir=model_dir)

m.fit(input_fn=train_input_fn, steps=2)

#
# m = tf.contrib.learn.KMeansClustering(2, model_dir=model_dir)
# m.fit(input_fn=train_input_fn, steps = 2)

results = m.evaluate(input_fn=eval_input_fn, steps=1)
print(results);
for key in sorted(results):
    print("%s: %s" % (key, results[key]))







