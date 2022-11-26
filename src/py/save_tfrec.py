import tensorflow as tf
import numpy as np
import pyarrow
from pyarrow import parquet as pqt
import glob
import pandas

files = glob.glob('model/train.parquet' + '/**/part-*.parquet', recursive=True)
ds = pqt.ParquetFile(files[0])
pd = ds.read().to_pandas()

cvec = tf.convert_to_tensor(np.vstack(pd['cvec'].to_numpy()))
lvec = tf.convert_to_tensor(np.vstack(pd['lvec'].to_numpy()))
lidnum = tf.convert_to_tensor(pd['lid_num'].to_numpy().astype('int32'))
output = tf.convert_to_tensor(pd['output'].to_numpy().astype('int32'))
ds = tf.data.Dataset.from_tensor_slices((cvec,lvec,lidnum,output))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return _bytes_feature(array)

def serialize_example(content, label, lid, output):
  
  # define the dictionary -- the structure -- of our single example
  data = {
    'content' : serialize_array(content),
    'label' : serialize_array(label),
    'lid' : _int64_feature(lid),
    'output' : _int64_feature(output)
  }
  
  # create an Example, wrapping the single features
  out = tf.train.Example(features=tf.train.Features(feature=data))
  return out.SerializeToString()

def tf_serialize_example(f0,f1,f2,f3):
  tf_string = tf.py_function(
    serialize_example,
    (f0, f1, f2, f3),  # Pass these args to the above function.
    tf.string)      # The return type is `tf.string`.
  return tf.reshape(tf_string, ()) # The result is a scalar.

ds2 = ds.map(tf_serialize_example)

filename = 'test.tfrecord'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(ds2)
writer.close()


instance_description = {
    'cvec': tf.io.FixedLenFeature([], tf.string),
    'lvec': tf.io.FixedLenFeature([], tf.string),
    'lidnum': tf.io.FixedLenFeature([], tf.int64),
    'output': tf.io.FixedLenFeature([], tf.int64)
}

def parse_instance(example):
  return tf.io.parse_single_example(example, instance_description)

ds3 = ds2.map(parse_instance)