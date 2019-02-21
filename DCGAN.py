import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data


def generator(z, reuse=False):
    with tf.variable_scope("G", reuse=reuse) as vs:
        conv1 = tf.layers.conv2d_transpose(z, 1024, [4, 4], strides=(1, 1), padding='valid',
                                           activation=tf.nn.leaky_relu, kernel_regularizer=tf.layers.batch_normalization)
        conv2 = tf.layers.conv2d_transpose(conv1, 512, [4, 4], strides=(2, 2), padding='same',
                                           activation=tf.nn.leaky_relu, kernel_regularizer=tf.layers.batch_normalization)
        conv3 = tf.layers.conv2d_transpose(conv2, 256, [4, 4], strides=(2, 2), padding='same',
                                           activation=tf.nn.leaky_relu, kernel_regularizer=tf.layers.batch_normalization)
        conv4 = tf.layers.conv2d_transpose(conv3, 128, [4, 4], strides=(2, 2), padding='same',
                                           activation=tf.nn.leaky_relu, kernel_regularizer=tf.layers.batch_normalization)
        conv5 = tf.layers.conv2d_transpose(conv4, 1, [4, 4], strides=(2, 2), padding='same')
        o = tf.nn.tanh(conv5)
    variables = tf.contrib.framework.get_variables(vs)
    return o, variables


def discriminator(x, reuse=False):
    with tf.variable_scope("D", reuse=reuse) as vs:
        conv1 = tf.layers.conv2d(x, 128, [4, 4], strides=(2, 2), padding='same',
                                 activation=tf.nn.leaky_relu, kernel_regularizer=tf.layers.batch_normalization)
        conv2 = tf.layers.conv2d(conv1, 256, [4, 4], strides=(2, 2), padding='same',
                                 activation=tf.nn.leaky_relu, kernel_regularizer=tf.layers.batch_normalization)
        conv3 = tf.layers.conv2d(conv2, 512, [4, 4], strides=(2, 2), padding='same',
                                 activation=tf.nn.leaky_relu, kernel_regularizer=tf.layers.batch_normalization)
        conv4 = tf.layers.conv2d(conv3, 1024, [4, 4], strides=(2, 2), padding='same',
                                 activation=tf.nn.leaky_relu, kernel_regularizer=tf.layers.batch_normalization)
        conv5 = tf.layers.conv2d(conv4, 1, [4, 4], strides=(1, 1), padding='valid')
        o = tf.nn.sigmoid(conv5)
    variables = tf.contrib.framework.get_variables(vs)
    return conv5, variables, o


def optimizer(loss, var_list, learning_rate=0.0002):
    optimizer_2 = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(
        loss,
        var_list=var_list
    )
    return optimizer_2


def main():

    if not os.path.exists('gen_img/'):
        os.makedirs('gen_img/')

    inp = tf.placeholder(tf.float32, [None, 64, 64, 1])
    random_inp = tf.placeholder(tf.float32, [None, 100])
    random_reshape = tf.reshape(random_inp, [-1, 1, 1, 100])

    generated_sample, g_var = generator(random_reshape)
    d_gen, d_var, d_gen_sig = discriminator(generated_sample)
    d_inp, d_var, d_inp_sig = discriminator(inp, True)

    d_loss_inp = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_inp, labels=tf.ones_like(d_inp)))
    d_loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_gen, labels=tf.zeros_like(d_gen)))

    d_loss = d_loss_gen + d_loss_inp
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_gen, labels=tf.ones_like(d_gen)))

    t_vars = tf.trainable_variables()
    dd_var = [var for var in t_vars if 'D' in var.name]
    gg_var = [var for var in t_vars if 'G' in var.name]

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_optim = optimizer(d_loss, var_list=dd_var)
        g_optim = optimizer(g_loss, var_list=gg_var)

    init = tf.global_variables_initializer()

    batch_size = 128

    with tf.Session() as sess:
        init.run()
        mnist = input_data.read_data_sets('/tmp/data/MNIST_data', one_hot=True)
        train_set = tf.image.resize_images(mnist.train.images.reshape([-1, 28, 28, 1]), [64, 64]).eval()
        train_set = (train_set - 0.5) / 0.5
        for step in range(1000):
            for ite in range(mnist.train.num_examples // batch_size):
                inp_np = train_set[ite*batch_size:(ite+1)*batch_size]
                random_np = np.random.uniform(-1., 1., size=[batch_size, 100])
                _, d_loss_p = sess.run([d_optim, d_loss], feed_dict={inp: inp_np, random_inp: random_np})
                random_np = np.random.uniform(-1., 1., size=[batch_size, 100])
                _, g_loss_p = sess.run([g_optim, g_loss], feed_dict={inp: inp_np, random_inp: random_np})
                print(str(step) + '/' + str(ite))
                print('d_loss: ' + str(d_loss_p))
                print('g_loss: ' + str(g_loss_p))
            random_np = np.random.uniform(-1., 1., size=[batch_size, 100])
            p_sample = sess.run(generated_sample, feed_dict={random_inp: random_np})
            p_sample = ((np.repeat(p_sample, 3, axis=3) + 1) * 127)
            p_sample = p_sample.astype('uint8')
            p_save = p_sample[:10].reshape([10*64, 64, 3])
            im = Image.fromarray(p_save)
            im.save("gen_img/" + str(step) + ".jpg")


if __name__ == '__main__':
    main()
