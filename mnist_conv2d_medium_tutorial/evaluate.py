import argparse
import tensorflow as tf

import mnist as mnist
from model import Model


def evaluate(args):
    images, labels = mnist.load_test_data(args.test_data)
    model = Model()

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint(args.checkpoint_file_path))

    logits = model(images, training=False)
    accuracy = model.accuracy(logits, labels)

    print('Test accuracy: {}'.format(accuracy.numpy()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_file_path', type=str, default='checkpoints', help='path to checkpoint directory')
    parser.add_argument('--test_data', type=str, default='data/mnist_test.csv', help='path to test data')
    args = parser.parse_args()

    evaluate(args)


if __name__ == '__main__':
    main()