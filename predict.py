import argparse
import tensorflow as tf
from get_model import model_builder
from helpers import generate_dataset


def main():
    filenames = tf.data.Dataset.list_files(args.data_dir)
    data = generate_dataset(filenames, args.batch_size, args.input_img_shape)

    model = model_builder(args.input_img_shape)
    model.load_weights(args.checkpoint_dir)
    eval_loss, eval_acc = model.evaluate(data)
    print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/cifar10/3.0.2/cifar10-test.tfrecord*',
                        help='Path to .tfrecord file, or file pattern for multiple files.')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/training',
                        help='Directory of training checkpoints')
    parser.add_argument('--input_img_shape', type=tuple, default=(32, 32, 3), help='Input img shape, ex: (32, 32, 3)')
    parser.add_argument('--batch_size', type=int, default=128, help='Global batch size')
    args = parser.parse_args()
    main()
