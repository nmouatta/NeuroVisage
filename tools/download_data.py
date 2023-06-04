import tensorflow_datasets as tfds

# Download data available through https://www.tensorflow.org/datasets/catalog/overview
# The data is downloaded in .tfrecord format 
builder = tfds.builder('cifar10', data_dir='~/NeuroVisage/data/')
builder.download_and_prepare()

