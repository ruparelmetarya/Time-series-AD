import tempfile
import urllib
import pandas as pd
import tensorflow as tf


"""Downloading data """
train_file = tempfile.NamedTemporaryFile()
test_file = tempfile.NamedTemporaryFile()
urllib.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)
urllib.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.test", test_file.name)



"""Reading files into pandas dataframes"""
COLUMNS = ["cpu", "memory", "temp", "power", "io", "smartAttribs", "network"]


# readcsv returns a data frame
df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)



LABEL_COLUMN = "label"




"""GROUPING COLUMNS: CATEGORICAL AND CONTINUOUS"""
CATEGORICAL_COLUMNS = []
CONTINUOUS_COLUMNS = ["cpu", "memory", "temp", "power", "io", "smartAttribs", "network"]




"""CONVERTING DATA INTO TENSORS"""

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
    label = tf.constant(df[LABEL_COLUMN].values)
    # Returns the feature columns and the label.
    return feature_cols, label


def train_input_fn():
    return input_fn(df_train)


def eval_input_fn():
    return input_fn(df_test)



#Similarly, we can define a RealValuedColumn for each continuous feature column that we want to use in the model:

cpu = tf.contrib.layers.real_valued_column("cpu")
memory = tf.contrib.layers.real_valued_column("memory")
temp = tf.contrib.layers.real_valued_column("temp")
power = tf.contrib.layers.real_valued_column("power")
io = tf.contrib.layers.real_valued_column("io")
smartAttribs = tf.contrib.layers.real_valued_column("smartAttribs")
network = tf.contrib.layers.real_valued_column("network")


model_dir = tempfile.mkdtemp()

m = tf.contrib.learn.LinearClassifier(feature_columns=[cpu, memory, temp, power, io, smartAttribs, network],
  model_dir=model_dir)


m.fit(input_fn=train_input_fn, steps=200)

results = m.evaluate(input_fn=eval_input_fn, steps=1)
print(results);
for key in sorted(results):
    print("%s: %s" % (key, results[key]))


