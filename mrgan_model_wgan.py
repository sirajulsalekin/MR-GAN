from ops_wgan import *


class Model:
    
    def __init__(self, batch_size, slope, num_pieces, lr, beta1, beta2, z_dim, raw_marginal):
        
        self.batch_size = batch_size
        self.slope = slope
        self.num_pieces = num_pieces
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.z_dim = z_dim
        self.raw_marginal = raw_marginal

        self.create_model()

    def create_model(self):

        self.input_x = tf.placeholder(tf.float32, shape=[None, 51, 4, 1])
        self.input_z = tf.placeholder(tf.float32, shape=[None, 1, 1, self.z_dim])

        self.train_g = tf.placeholder(tf.bool, [])
        self.train_d = tf.placeholder(tf.bool, [])

        self.keep_prob = tf.placeholder(tf.float32)

        self.G_x = self.decoder(self.input_z, train=self.train_g, reuse=False)
        self.G_z = self.encoder(self.input_x, train=self.train_g, reuse=False)

        self.resampler = self.decoder(self.encoder(self.input_x, train=False, reuse=True), train=False, reuse=True)

        self.D_G_x, self.disc_fake = self.discriminator(self.G_x, self.input_z, self.keep_prob, train=self.train_d, reuse=False)
        self.D_G_z, self.disc_real = self.discriminator(self.input_x, self.G_z, self.keep_prob, train=self.train_d, reuse=True)

        self.g_loss = -tf.reduce_mean(self.disc_fake)
        self.d_loss = tf.reduce_mean(self.disc_fake) - tf.reduce_mean(self.disc_real)

        # Gradient penalty
        self.alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        self.differences_x = self.G_x - self.input_x
        self.differences_z = self.input_z - self.G_z
        self.interpolates_x = self.input_x + (self.alpha*self.differences_x)
        self.interpolates_z = self.G_z + (self.alpha*self.differences_z)
        _, self.disc_interpolate = self.discriminator(self.interpolates_x, self.interpolates_z, 1, train=self.train_d, reuse=True)

        self.gradients_x = tf.gradients(self.disc_interpolate, self.interpolates_x)[0]
        self.slopes_x = tf.sqrt(tf.reduce_sum(tf.square(self.gradients_x), reduction_indices=[1]))
        self.gradients_z = tf.gradients(self.disc_interpolate, self.interpolates_z)[0]
        self.slopes_z = tf.sqrt(tf.reduce_sum(tf.square(self.gradients_z), reduction_indices=[1]))
        self.gradient_penalty = tf.reduce_mean((self.slopes_x-1.)**2) + tf.reduce_mean((self.slopes_z-1.)**2)
        self.d_loss += 10*self.gradient_penalty
        
        # self.differences_x_ = tf.reduce_mean(self.G_x)
        # self.differences_z_ = tf.reduce_mean(self.G_z)

        self.d_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        self.gx_vars = [var for var in tf.trainable_variables() if 'decoder' in var.name]
        self.gz_vars = [var for var in tf.trainable_variables() if 'encoder' in var.name]

        self.d_optim = tf.train.AdamOptimizer(self.lr, self.beta1, self.beta2).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, self.beta1, self.beta2).minimize(self.g_loss, var_list=self.gx_vars+self.gz_vars)

        self.optims = [self.d_optim, self.g_optim]

        print('----- Discriminator Variables -----')
        for var in self.d_vars:
            print(var.name)
        print('----- Discriminator Variables -----')
        print('----- Generator X Variables -----')
        for var in self.gx_vars:
            print(var.name)
        print('----- Generator X Variables -----')
        print('----- Generator Z Variables -----')
        for var in self.gz_vars:
            print(var.name)
        print('----- Generator Z Variables -----')

    def decoder(self, input_, train=True, reuse=False):

        with tf.variable_scope('decoder', reuse=reuse) as scope:

            h = input_
            # input, kernel_shape, c_d_shape, deconv_shape, step_size)
            with tf.variable_scope('layer1'):
                h = deconv(h, [4, 1, 256, 64], [self.batch_size, 1, 1, 64], [self.batch_size, 4, 1, 256], [1, 1, 1, 1])
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=self.slope)

            with tf.variable_scope('layer2'):
                h = deconv(h, [5, 1, 128, 256], [self.batch_size, 4, 1, 256], [self.batch_size, 11, 1, 128], [1, 2, 2, 1])
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=self.slope)

            with tf.variable_scope('layer3'):
                h = deconv(h, [6, 1, 64, 128], [self.batch_size, 11, 1, 128], [self.batch_size, 16, 1, 64], [1, 1, 1, 1])
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=self.slope)

            with tf.variable_scope('layer4'):
                h = deconv(h, [6, 1, 32, 64], [self.batch_size, 16, 1, 64], [self.batch_size, 36, 1, 32], [1, 2, 2, 1])
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=self.slope)

            with tf.variable_scope('layer5'):
                h = deconv(h, [16, 4, 32, 32], [self.batch_size, 36, 1, 32], [self.batch_size, 51, 4, 32], [1, 1, 1, 1])
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=self.slope)

            with tf.variable_scope('layer6'):
                h = conv2d(h, [1, 1, 32, 32], [1, 1, 1, 1])
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=self.slope)

            with tf.variable_scope('layer7'):
                marginal = cal_marginal(self.raw_marginal)
                h = conv2d(h, [1, 1, 32, 1], [1, 1, 1, 1])
                # h = add_nontied_bias(h, initializer=tf.constant_initializer(marginal))
                h = tf.tanh(h)  # tf.nn.sigmoid(h)

            output = h

        return output

    def encoder(self, input_, train=True, reuse=False):

        input = input_

        with tf.variable_scope('encoder', reuse=reuse) as scope:

            h = input

            with tf.variable_scope('layer1'):  # 15
                h = conv2d(h, [16, 4, 1, 32], [1, 1, 1, 1])
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=self.slope)

            with tf.variable_scope('layer2'):  # 7
                h = conv2d(h, [6, 1, 32, 64], [1, 2, 2, 1])
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=self.slope)

            with tf.variable_scope('layer3'):  # 5
                h = conv2d(h, [6, 1, 64, 128], [1, 1, 1, 1])
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=self.slope)

            with tf.variable_scope('layer4'):  # 2
                h = conv2d(h, [5, 1, 128, 256], [1, 2, 2, 1])
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=self.slope)

            with tf.variable_scope('layer5'):  # 1
                h = conv2d(h, [4, 1, 256, 512], [1, 1, 1, 1])
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=self.slope)

            with tf.variable_scope('layer6'):
                h = conv2d(h, [1, 1, 512, 512], [1, 1, 1, 1])
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=self.slope)

            with tf.variable_scope('layer7'):
                h = conv2d(h, [1, 1, 512, 64], [1, 1, 1, 1])
                # h = batch_norm(h, training=train)
            """
            with tf.variable_scope('layer7_mu'):
                h_mu = conv2d(h, [1, 1, 512, 64], [1, 1, 1, 1])
                h_mu = add_nontied_bias(h_mu)
                G_z_mu = h_mu

            with tf.variable_scope('layer7_sigma'):
                h_sigma = conv2d(h, [1, 1, 512, 64], [1, 1, 1, 1])
                h_sigma = add_nontied_bias(h_sigma)
                G_z_sigma = h_sigma
                h_sigma = tf.exp(h_sigma)

            rng = tf.random_normal(shape=tf.shape(h_mu))
            """
            output = tf.tanh(h)  # (rng * h_sigma) + h_mu

        return output

    def discriminator(self, input_x, input_z, keep_prob, train=True, reuse=False):

        with tf.variable_scope('discriminator', reuse=reuse):

            h_x = input_x
            h_z = input_z

            with tf.variable_scope('x'):

                with tf.variable_scope('layer1'):
                    # h_x = tf.nn.dropout(h_x, keep_prob)
                    h_x = conv2d(h_x, [16, 4, 1, 32], [1, 1, 1, 1])
                    h_x = lrelu(h_x, slope=self.slope)  # conv_maxout(h_x, self.num_pieces)

                with tf.variable_scope('layer2'):
                    # h_x = tf.nn.dropout(h_x, keep_prob)
                    h_x = conv2d(h_x, [6, 1, 32, 64], [1, 2, 2, 1])  # conv2d(h_x, [3, 1, 16, 64], [1, 2, 2, 1])
                    h_x = lrelu(h_x, slope=self.slope)

                with tf.variable_scope('layer3'):
                    # h_x = tf.nn.dropout(h_x, keep_prob)
                    h_x = conv2d(h_x, [6, 1, 64, 128], [1, 1, 1, 1])
                    h_x = lrelu(h_x, slope=self.slope)

                with tf.variable_scope('layer4'):
                    # h_x = tf.nn.dropout(h_x, keep_prob)
                    h_x = conv2d(h_x, [5, 1, 128, 256], [1, 2, 2, 1])
                    h_x = lrelu(h_x, slope=self.slope)

                with tf.variable_scope('layer5'):
                    # h_x = tf.nn.dropout(h_x, keep_prob)
                    h_x = conv2d(h_x, [4, 1, 256, 512], [1, 1, 1, 1])
                    h_x = lrelu(h_x, slope=self.slope)  # conv_maxout was not here

            with tf.variable_scope('z'):

                with tf.variable_scope('layer1'):
                    # h_z = tf.nn.dropout(h_z, keep_prob)
                    h_z = conv2d(h_z, [1, 1, 64, 512], [1, 1, 1, 1])
                    h_z = lrelu(h_z, slope=self.slope)

                with tf.variable_scope('layer2'):
                    # h_z = tf.nn.dropout(h_z, keep_prob)
                    h_z = conv2d(h_z, [1, 1, 512, 512], [1, 1, 1, 1])
                    h_z = lrelu(h_z, slope=self.slope)

            with tf.variable_scope('xz'):

                h_xz = tf.concat(3, [h_x, h_z])  # h_x.get_shape().ndims-1)

                with tf.variable_scope('layer1'):
                    # h_xz = tf.nn.dropout(h_xz, keep_prob)
                    h_xz = conv2d(h_xz, [1, 1, 1024, 1024], [1, 1, 1, 1])
                    h_xz = lrelu(h_xz, slope=self.slope)

                with tf.variable_scope('layer2'):
                    # h_xz = tf.nn.dropout(h_xz, keep_prob)
                    h_xz = conv2d(h_xz, [1, 1, 1024, 1024], [1, 1, 1, 1])
                    h_xz = lrelu(h_xz, slope=self.slope)

                with tf.variable_scope('layer3'):
                    # h_xz = tf.nn.dropout(h_xz, keep_prob)
                    h_xz = conv2d(h_xz, [1, 1, 1024, 1], [1, 1, 1, 1])

            logits = h_xz
            output = tf.nn.sigmoid(h_xz)

            return output, logits
