# Importing modules
import argparse
import os
import tensorflow as tf
import numpy as np
import tables as tb
import glob
import utils
from GFL_model_wgan import Model

parser = argparse.ArgumentParser()
parser.add_argument("--h5_dir", help="Directory of saved HDF5 file")
parser.add_argument("--log_dir", help="Log directory")
args = parser.parse_args()
os.mkdir(args.log_dir + 'image/')
os.mkdir(args.log_dir + 'model/')

h5_files = args.h5_dir + "*.h5"

tf.reset_default_graph()
writer = tf.summary.FileWriter(args.log_dir)

slope = 0.2  # for leaky relu
num_pieces = 2  # for convolution max-out
z_dim = 64  # dimension of noise sample
image_shape = (51, 4, 1)  # (17, 100, 1)
num_epoch = 2
batch_size = 100
lr, beta1, beta2 = 5e-3, 0.5, 0.9

h5 = tb.open_file(args.h5_dir + "chr22.h5", 'r')  # (args.h5_dir + "SE.h5", 'r')  #
data_ = h5.root.data
raw_marginal = data_[0:500, :]
validation_x = data_[500:5600].reshape(batch_size, 51, 4, 1)
# validation_x = 2*((validation_x_-np.min(validation_x_))/(np.max(validation_x_)-np.min(validation_x_)))-1
utils.save_images(args.log_dir + 'image/valid_input.png', validation_x, size=(4, 25))  # validation_x_, (25, 4)

# Load model
gfl = Model(batch_size, slope, num_pieces, lr, beta1, beta2, z_dim, raw_marginal)
saver = tf.train.Saver(max_to_keep=30)
# Summaries
tf.summary.image('input_x', gfl.input_x, max_outputs=10)
tf.summary.image('generated_x', gfl.G_x, max_outputs=20)
tf.summary.image('resampled_x', gfl.resampler, max_outputs=20)
tf.summary.scalar('Discriminator loss', gfl.d_loss)
tf.summary.scalar('Generator loss', gfl.g_loss)

for var in (gfl.gx_vars + gfl.gz_vars + gfl.d_vars):
    tf.summary.histogram(var.name, var)

summ = tf.summary.merge_all()
# summ_Dloss = tf.summary.scalar('Discriminator loss', gfl.d_loss)
# summ_Gloss = tf.summary.scalar('Generator loss', gfl.g_loss)
# summ = tf.summary.merge([summ_Dloss, summ_Gloss])

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
            print('Started {}, Total samples {}'.format(genome_filename.replace(args.h5_dir, ''), str(data_size//51)))  # (genome_filename, len(da_ta))
            # Train
            perm = np.random.permutation(data_size // (batch_size*51))
            for idx in perm:
                for k in range(3):  # Train generator 10 times for one training of discriminator
                    # Train generator
                    batch_x = (da_ta[idx*batch_size*51:(idx+1)*batch_size*51]).reshape(batch_size, 51, 4, 1)
                    # batch_x = 2*((batch_x_-np.min(batch_x_))/(np.max(batch_x_)-np.min(batch_x_)))-1
                    batch_z = np.random.normal(size=(batch_size, 1, 1, z_dim)).astype(np.float32)
                    feeds = {gfl.input_x: batch_x, gfl.input_z: batch_z, gfl.train_g: True, gfl.train_d: True, gfl.keep_prob: 0.5}
                    sess.run(gfl.d_optim, feed_dict=feeds)
                # Train discriminator
                sess.run(gfl.g_optim, feed_dict=feeds)
                global_step += 1

                if global_step == 1:
                    saver.save(sess, args.log_dir + 'model/wgan_gp', global_step=global_step)

                elif global_step % 200 == 0:
                    validation_z = np.random.normal(size=(batch_size, 1, 1, z_dim)).astype(np.float32)
                    feeds = {gfl.input_x: validation_x, gfl.input_z: validation_z, gfl.train_g: False, gfl.train_d: False, gfl.keep_prob: 1.0}
                    summ_, d_loss_, g_loss_, samples, resamples = sess.run([summ, gfl.d_loss, gfl.g_loss, gfl.G_x, gfl.resampler], feed_dict=feeds)
                    writer.add_summary(summ_, global_step)
                    print("step %d, d_loss %g, g_loss %g" % (global_step, d_loss_, g_loss_))
                    if (-30 <= d_loss_ <= 0) or (0 <= g_loss_ <= 50):
                        utils.save_images(args.log_dir + 'image/step_{}_{}_{}.png'.format(str(global_step), str(int(d_loss_)), str(int(g_loss_))), samples, size=(4, 25))
                        utils.save_images(args.log_dir + 'image/recons_{}_{}_{}.png'.format(str(global_step), str(int(d_loss_)), str(int(g_loss_))), resamples, size=(4, 25))
                        saver.save(sess, args.log_dir + 'model/wgan_gp', global_step=global_step, write_meta_graph=False)


                    # Validation

                    """
                    if samples.shape[-1] == 1:
                        broad = np.zeros(samples.shape[:-1] + (3,)).astype(np.float32)
                        broad += samples
                        samples = broad

                    utils.save_images(args.log_dir + 'image/step_{}.png'.format(str(global_step)), samples)
                    utils.save_images(args.log_dir + 'image/reconst_step_{}.png'.format(str(global_step)), resamples)
                    """
