import tensorflow as tf


def _read_tfrecord(serialized_example, img_size):
    feature_description = {
        "image": tf.io.FixedLenFeature((), tf.string, ""),
        "label": tf.io.FixedLenFeature((), tf.int64, -1),
    }
    example = tf.io.parse_single_example(
        serialized_example, feature_description
    )
    image = tf.io.decode_jpeg(example['image'], channels=3)
    image = tf.image.resize(image, (img_size[0], img_size[1]))
    image = tf.cast(image, tf.float32)
    image = image / 255
    return image, example['label']


def generate_dataset(files, batch_size, img_size):
    dataset = files.interleave(
        tf.data.TFRecordDataset,
        cycle_length=tf.data.experimental.AUTOTUNE,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    dataset = dataset.map(lambda x: _read_tfrecord(x, img_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(len(files))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
