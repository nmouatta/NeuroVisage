import os
import argparse
import tensorflow as tf
from get_model import model_builder
from helpers import generate_dataset


def main():
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(args.checkpoint_dir, 'training/cp-{'
                                                                                               'epoch:03d}.ckpt'),
                                                             save_best_only=True, save_weights_only=True)
    learning_rate_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    training_files = tf.data.Dataset.list_files(args.train_input_dir)
    validation_files = tf.data.Dataset.list_files(args.val_input_dir)

    train_dataset = generate_dataset(training_files, args.batch_size, args.input_img_shape)
    val_dataset = generate_dataset(validation_files, args.batch_size, args.input_img_shape)

    model = model_builder(args.input_img_shape)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                  metrics=['accuracy'])
    model.fit(train_dataset, validation_data=val_dataset, epochs=args.num_epochs,
              callbacks=[checkpoint_callback, learning_rate_callback])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_input_dir', type=str, default='./data/cifar10/3.0.2/cifar10-train.tfrecord*',
                        help='Path to .tfrecord training file or file pattern for multiple files.')
    parser.add_argument('--val_input_dir', type=str, default='./data/cifar10/3.0.2/cifar10-test.tfrecord*',
                        help='Path to .tfrecord validation file or file pattern for multiple files.')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory path for saving training checkpoints')
    parser.add_argument('--input_img_shape', type=tuple, default=(32, 32, 3), help='Input img shape, ex: (32, 32, 3)')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Global batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning Rate')
    args = parser.parse_args()
    main()
