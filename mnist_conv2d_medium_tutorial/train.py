import argparse
import tensorflow as tf

import mnist as mnist
from model import Model

NUM_LABELS = 10


def train(args):
    model = Model()
    images, val_images, labels, val_labels = mnist.load_train_data(args.train_data)

    optimizer = tf.keras.optimizers.Adam()
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, args.checkpoint_file_path, max_to_keep=1)

    writer = tf.summary.create_file_writer(args.summary_dir)

    for i in range(args.num_iter):
        offset = (i * args.batch_size) % (len(images) - args.batch_size)
        batch_x, batch_y = images[offset:(offset + args.batch_size), :], labels[offset:(offset + args.batch_size), :]

        with tf.GradientTape() as tape:
            logits = model(batch_x, training=True)
            loss = model.compute_loss(logits=logits, labels=batch_y)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        with writer.as_default():
            tf.summary.scalar('loss', loss, step=i)

        print(i, loss.numpy())

        if i % 1000 == 0:
            val_logits = model(val_images, training=False)
            accuracy = model.accuracy(val_logits, val_labels)
            print('Iter {} Accuracy: {}'.format(i, accuracy.numpy()))

        if i == args.num_iter - 1:
            checkpoint_manager.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='size of training batches')
    parser.add_argument('--num_iter', type=int, default=10000, help='number of training iterations')
    parser.add_argument('--checkpoint_file_path', type=str, default='checkpoints/model', help='path to checkpoint directory')
    parser.add_argument('--train_data', type=str, default='data/mnist_train.csv', help='path to train and test data')
    parser.add_argument('--summary_dir', type=str, default='graphs', help='path to directory for storing summaries')
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()