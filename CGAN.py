import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data


# generate image
def generator(z, label, reuse=False):
    with tf.variable_scope("G", reuse=reuse) as vs:
        z_concat = tf.concat([z, label], 3)
        conv1 = tf.layers.conv2d_transpose(z_concat, 1024, [4, 4], strides=(1, 1), padding='valid',
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


# discriminate image
def discriminator(x, label_map, reuse=False):
    with tf.variable_scope("D", reuse=reuse) as vs:
        x_concat = tf.concat([x, label_map], 3)
        conv1 = tf.layers.conv2d(x_concat, 128, [4, 4], strides=(2, 2), padding='same',
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


def main():
    # make folder
    if not os.path.exists('acgan_img/'):
        os.makedirs('acgan_img/')

    # input placeholder
    set_inp = tf.placeholder(tf.float32, [None, 64, 64, 1])
    label_inp = tf.placeholder(tf.float32, [None, 10])
    label_reshape = tf.reshape(label_inp, [-1, 1, 1, 10])
    label_map_inp = tf.placeholder(tf.float32, [None, 64, 64, 10])
    random_inp = tf.placeholder(tf.float32, [None, 100])
    random_reshape = tf.reshape(random_inp, [-1, 1, 1, 100])

    # generate image
    generated_sample, g_var = generator(random_reshape, label_reshape)

    # discriminate real and fake
    d_gen, d_var, d_gen_sig = discriminator(generated_sample, label_map_inp)
    d_inp, d_var, d_inp_sig = discriminator(set_inp, label_map_inp, True)

    # loss for input image and generated image
    d_loss_inp = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_inp, labels=tf.ones_like(d_inp)))
    d_loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_gen, labels=tf.zeros_like(d_gen)))

    # loss for discriminator and generator
    d_loss = d_loss_gen + d_loss_inp
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_gen, labels=tf.ones_like(d_gen)))

    # variables for discriminator and generator
    t_vars = tf.trainable_variables()
    dd_var = [var for var in t_vars if 'D' in var.name]
    gg_var = [var for var in t_vars if 'G' in var.name]

    # optimize discriminator and generator
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_optim = tf.train.AdamOptimizer(2e-6, beta1=0.5).minimize(d_loss, var_list=dd_var)
        g_optim = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(g_loss, var_list=gg_var)

    init = tf.global_variables_initializer()

    batch_size = 128

    with tf.Session() as sess:
        init.run()
        # load mnist data
        mnist = input_data.read_data_sets('/tmp/data/MNIST_data', one_hot=True)
        train_label = mnist.train.labels

        # resize and normalize image
        train_set = tf.image.resize_images(mnist.train.images.reshape([-1, 28, 28, 1]), [64, 64]).eval()
        train_set = (train_set - 0.5) / 0.5

        # training network
        for step in range(1000):
            for ite in range(mnist.train.num_examples // batch_size):
                set_np = train_set[ite*batch_size:(ite+1)*batch_size]
                label_np = train_label[ite * batch_size:(ite + 1) * batch_size]
                label_map_np = np.reshape(label_np, (batch_size, 1, 1, 10))
                label_map_np = np.repeat(np.repeat(label_map_np, 64, axis=1), 64, axis=2)
                random_np = np.random.uniform(-1., 1., size=[batch_size, 100])
                _, d_loss_p = sess.run([d_optim, d_loss], feed_dict={set_inp: set_np, random_inp: random_np,
                                                                     label_inp: label_np, label_map_inp: label_map_np})
                random_np = np.random.uniform(-1., 1., size=[batch_size, 100])
                _, g_loss_p = sess.run([g_optim, g_loss], feed_dict={set_inp: set_np, random_inp: random_np,
                                                                     label_inp: label_np, label_map_inp: label_map_np})
                print(str(step) + '/' + str(ite))
                print('d_loss: ' + str(d_loss_p))
                print('g_loss: ' + str(g_loss_p))

            # generate image
            random_np = np.random.uniform(-1., 1., size=[batch_size, 100])
            label_np = np.zeros((batch_size, 10))
            for i in range(batch_size):
                label_np[i, i % 10] = 1
            p_sample = sess.run(generated_sample, feed_dict={random_inp: random_np, label_inp: label_np})

            # save image
            p_sample = ((np.repeat(p_sample, 3, axis=3) + 1) * 127)
            p_sample = p_sample.astype('uint8')
            p_save = p_sample[:10].reshape([10*64, 64, 3])
            im = Image.fromarray(p_save)
            im.save("acgan_img/" + str(step) + ".jpg")


if __name__ == '__main__':
    main()