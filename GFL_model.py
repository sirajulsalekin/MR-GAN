from ops import *


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

        self.input_x = tf.placeholder(tf.float32, shape=[None, 17, 100, 1])
        self.input_z = tf.placeholder(tf.float32, shape=[None, 1, 1, self.z_dim])

        self.train_g = tf.placeholder(tf.bool, [])
        self.train_d = tf.placeholder(tf.bool, [])

        self.keep_prob = tf.placeholder(tf.float32)

        self.G_x = self.decoder(self.input_z, train=self.train_g, reuse=False)
        self.G_z = self.encoder(self.input_x, train=self.train_g, reuse=False)

        self.resampler = self.decoder(self.encoder(self.input_x, train=False, reuse=True), train=False, reuse=True)

        self.D_G_x, self.D_G_x_logits = self.discriminator(self.G_x, self.input_z, self.keep_prob, train=self.train_d, reuse=False)
        self.D_G_z, self.D_G_z_logits = self.discriminator(self.input_x, self.G_z, self.keep_prob, train=self.train_d, reuse=True)

        self.d_loss = tf.reduce_mean(tf.nn.softplus(-self.D_G_z_logits) + tf.nn.softplus(self.D_G_x_logits))
        self.g_loss = tf.reduce_mean(tf.nn.softplus(self.D_G_z_logits) + tf.nn.softplus(-self.D_G_x_logits))

        self.d_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        self.gx_vars = [var for var in tf.trainable_variables() if 'decoder' in var.name]
        self.gz_vars = [var for var in tf.trainable_variables() if 'encoder' in var.name]

        self.d_optim = tf.train.AdamOptimizer(self.lr, self.beta1, self.beta2).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, self.beta1, self.beta2).minimize(self.g_loss, var_list=self.gx_vars+self.gz_vars)

        self.optims = [self.d_optim, self.g_optim]

        return self.optims

    def decoder(self, input_, train=True, reuse=False):

        with tf.variable_scope('decoder', reuse=reuse) as scope:

            h = input_
            # input, kernel_shape, c_d_shape, deconv_shape, step_size)
            with tf.variable_scope('layer1'):
                h = deconv(h, [2, 1, 256, 64], [self.batch_size, 1, 1, 64], [self.batch_size, 2, 1, 256], [1, 1, 1, 1])
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=self.slope)

            with tf.variable_scope('layer2'):
                h = deconv(h, [3, 1, 128, 256], [self.batch_size, 2, 1, 256], [self.batch_size, 5, 1, 128], [1, 2, 2, 1])
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=self.slope)

            with tf.variable_scope('layer3'):
                h = deconv(h, [3, 1, 64, 128], [self.batch_size, 5, 1, 128], [self.batch_size, 7, 1, 64], [1, 1, 1, 1])
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=self.slope)

            with tf.variable_scope('layer4'):
                h = deconv(h, [3, 1, 32, 64], [self.batch_size, 7, 1, 64], [self.batch_size, 15, 1, 32], [1, 2, 2, 1])
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=self.slope)

            with tf.variable_scope('layer5'):
                h = deconv(h, [3, 100, 32, 32], [self.batch_size, 15, 1, 32], [self.batch_size, 17, 100, 32], [1, 1, 1, 1])
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=self.slope)

            with tf.variable_scope('layer6'):
                h = conv2d(h, [1, 1, 32, 32], [1, 1, 1, 1])
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=self.slope)

            with tf.variable_scope('layer7'):
                marginal = cal_marginal(self.raw_marginal)
                h = conv2d(h, [1, 1, 32, 1], [1, 1, 1, 1])
                h = add_nontied_bias(h, initializer=tf.constant_initializer(marginal))
                h = tf.nn.sigmoid(h)

            output = h

        return output

    def encoder(self, input_, train=True, reuse=False):

        input = input_

        with tf.variable_scope('encoder', reuse=reuse) as scope:

            h = input

            with tf.variable_scope('layer1'):  # 15
                h = conv2d(h, [3, 100, 1, 32], [1, 1, 1, 1])
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=self.slope)

            with tf.variable_scope('layer2'):  # 7
                h = conv2d(h, [3, 1, 32, 64], [1, 2, 2, 1])
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=self.slope)

            with tf.variable_scope('layer3'):  # 5
                h = conv2d(h, [3, 1, 64, 128], [1, 1, 1, 1])
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=self.slope)

            with tf.variable_scope('layer4'):  # 2
                h = conv2d(h, [3, 1, 128, 256], [1, 2, 2, 1])
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=self.slope)

            with tf.variable_scope('layer5'):  # 1
                h = conv2d(h, [2, 1, 256, 512], [1, 1, 1, 1])
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=self.slope)

            with tf.variable_scope('layer6'):
                h = conv2d(h, [1, 1, 512, 512], [1, 1, 1, 1])
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=self.slope)

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

            output = (rng * h_sigma) + h_mu

        return output

    def discriminator(self, input_x, input_z, keep_prob, train=True, reuse=False):

        with tf.variable_scope('discriminator', reuse=reuse):

            h_x = input_x
            h_z = input_z

            with tf.variable_scope('x'):

                with tf.variable_scope('layer1'):
                    h_x = tf.nn.dropout(h_x, keep_prob)
                    h_x = conv2d(h_x, [3, 100, 1, 32], [1, 1, 1, 1])
                    h_x = conv_maxout(h_x, self.num_pieces)

                with tf.variable_scope('layer2'):
                    h_x = tf.nn.dropout(h_x, keep_prob)
                    h_x = conv2d(h_x, [3, 1, 16, 64], [1, 2, 2, 1])
                    h_x = conv_maxout(h_x, self.num_pieces)

                with tf.variable_scope('layer3'):
                    h_x = tf.nn.dropout(h_x, keep_prob)
                    h_x = conv2d(h_x, [3, 1, 32, 128], [1, 1, 1, 1])
                    h_x = conv_maxout(h_x, self.num_pieces)

                with tf.variable_scope('layer4'):
                    h_x = tf.nn.dropout(h_x, keep_prob)
                    h_x = conv2d(h_x, [3, 1, 64, 256], [1, 2, 2, 1])
                    h_x = conv_maxout(h_x, self.num_pieces)

                with tf.variable_scope('layer5'):
                    h_x = tf.nn.dropout(h_x, keep_prob)
                    h_x = conv2d(h_x, [2, 1, 128, 512], [1, 1, 1, 1])
                    # h_x = conv_maxout(h_x, self.num_pieces)

            with tf.variable_scope('z'):

                with tf.variable_scope('layer1'):
                    h_z = tf.nn.dropout(h_z, keep_prob)
                    h_z = conv2d(h_z, [1, 1, 64, 512], [1, 1, 1, 1])
                    h_z = conv_maxout(h_z, self.num_pieces)

                with tf.variable_scope('layer2'):
                    h_z = tf.nn.dropout(h_z, keep_prob)
                    h_z = conv2d(h_z, [1, 1, 256, 512], [1, 1, 1, 1])
                    # h_z = conv_maxout(h_z, self.num_pieces)

            with tf.variable_scope('xz'):

                h_xz = tf.concat(3, [h_x, h_z])  # h_x.get_shape().ndims-1)

                with tf.variable_scope('layer1'):
                    h_xz = tf.nn.dropout(h_xz, keep_prob)
                    h_xz = conv2d(h_xz, [1, 1, 1024, 1024], [1, 1, 1, 1])
                    h_xz = conv_maxout(h_xz, self.num_pieces)

                with tf.variable_scope('layer2'):
                    h_xz = tf.nn.dropout(h_xz, keep_prob)
                    h_xz = conv2d(h_xz, [1, 1, 512, 1024], [1, 1, 1, 1])
                    h_xz = conv_maxout(h_xz, self.num_pieces)

                with tf.variable_scope('layer3'):
                    h_xz = tf.nn.dropout(h_xz, keep_prob)
                    h_xz = conv2d(h_xz, [1, 1, 512, 1], [1, 1, 1, 1])

            logits = h_xz
            output = tf.nn.sigmoid(h_xz)

            return output, logits
