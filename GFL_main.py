# Importing modules
import argparse
import tensorflow as tf
import numpy as np
import tables as tb
import glob
from GFL_model import Model

parser = argparse.ArgumentParser()
parser.add_argument("--h5_dir", help="Directory of saved HDF5 file")
parser.add_argument("--log_dir", help="Log directory")
args = parser.parse_args()

h5_files = args.h5_dir + "*.h5"

tf.reset_default_graph()
writer = tf.summary.FileWriter(args.log_dir)

slope = 0.2  # for leaky relu
num_pieces = 2  # for convolution max-out
z_dim = 64  # dimension of noise sample
image_shape = (17, 100, 1)
num_epoch = 5
batch_size = 100
lr = 1e-5

h5 = tb.open_file(args.h5_dir + "chr22.h5", 'r')
data_ = h5.root.data
raw_marginal = data_[0:500, :]
validation_x = data_[500:2200].reshape(batch_size, 17, 100, 1)

# Load model
gfl = Model(batch_size, slope, num_pieces, lr, z_dim, raw_marginal)
saver = tf.train.Saver()
# Summaries
summ_Dloss = tf.summary.scalar('Discriminator loss', gfl.d_loss)
summ_Gloss = tf.summary.scalar('Generator loss', gfl.g_loss)
summ = tf.summary.merge([summ_Dloss, summ_Gloss])

with tf.Session() as sess:
    writer.add_graph(sess.graph)
    tf.global_variables_initializer().run()

    global_step = 0

    for epoch in range(num_epoch):
        print("EPOCH %d" % (epoch))

        for genome_filename in glob.glob(h5_files):
            h5f = tb.open_file(genome_filename, 'r')
            da_ta = h5f.root.data
            data_size = len(da_ta)
            print('Started {}, Total samples {}'.format(genome_filename.replace(args.h5_dir, ''), str(data_size//17)))  # (genome_filename, len(da_ta))
            # Train
            perm = np.random.permutation(data_size // (batch_size*17))
            for idx in perm:
                for k in range(10):  # Train generator 10 times for one training of discriminator
                    # Train generator
                    batch_x = da_ta[idx*batch_size*17:(idx+1)*batch_size*17].reshape(batch_size, 17, 100, 1)
                    batch_z = np.random.normal(size=(batch_size, 1, 1, z_dim)).astype(np.float32)
                    feeds = {gfl.input_x: batch_x, gfl.input_z: batch_z, gfl.train_g: True, gfl.train_d: True, gfl.keep_prob: 0.5}
                    sess.run(gfl.g_optim, feed_dict=feeds)
                # Train discriminator
                sess.run(gfl.d_optim, feed_dict=feeds)
                global_step += 1

                if global_step % 1000 == 1:
                    if global_step == 1:
                        saver.save(sess, args.log_dir + 'gfl_model', global_step=global_step)
                    # Validation
                    validation_z = np.random.normal(size=(batch_size, 1, 1, z_dim)).astype(np.float32)
                    feeds = {gfl.input_x: validation_x, gfl.input_z: validation_z, gfl.train_g: False, gfl.train_d: False, gfl.keep_prob: 1.0}
                    summ_, d_loss_, g_loss_ = sess.run([summ, gfl.d_loss, gfl.g_loss], feed_dict=feeds)
                    writer.add_summary(summ_, global_step)
                    saver.save(sess, args.log_dir + 'gfl_model', global_step=global_step, write_meta_graph=False)
                    print("step %d, d_loss %g, g_loss %g" % (global_step, d_loss_, g_loss_))
