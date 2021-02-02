import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time 

import tensorflow as tf
import img_read as utils
from __future__ import print_function

class ConvNet(object):
    def __init__(self):
        self.lr = 0.001
        self.batch_size = 128
        self.keep_prob = tf.constant(0.75)
        self.gstep = tf.Variable(0, dtype=tf.int32, 
                                trainable=False, name='global_step')
        self.n_classes = 4
        self.skip_step = 20
        self.n_test = 25
        self.training=False

    def get_data(self):
        with tf.name_scope('data'):
            train_data, test_data = utils.get_data()
            iterator = tf.data.Iterator.from_structure(train_data.output_types, 
                                                   train_data.output_shapes)
            img, self.label = iterator.get_next()
            print(img.shape)
            _, nf, r, c, ch = img.shape
            self.img = tf.reshape(img, shape=[1, nf, r, c, ch])
            # reshape the image to make it work with tf.nn.conv2d

            self.train_init = iterator.make_initializer(train_data)  # initializer for train_data
            self.test_init = iterator.make_initializer(test_data)    # initializer for train_data

    def inference(self):
        img = tf.cast(self.img, tf.float32)
        conv1 = tf.layers.conv3d(inputs=img,
                                  filters=5,
                                  kernel_size=[1, 5, 5],
                                  padding='same',
                                  activation=tf.nn.relu,
                                  data_format='channels_last',
                                  name='conv1')

        pool1 = tf.layers.max_pooling3d(inputs=conv1, 
                                                pool_size=[1, 2, 2], 
                                                strides=[1, 2, 2], 
                                                data_format='channels_last',
                                                name='pool1')

        conv2 = tf.layers.conv3d(inputs=pool1,
                                          filters=5,
                                          kernel_size=[1, 5, 5],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          data_format='channels_last',
                                          name='conv2')

        pool2 = tf.layers.max_pooling3d(inputs=conv2, 
                                                pool_size=[1, 2, 2], 
                                                strides=[1, 2, 2], 
                                                data_format='channels_last',
                                                name='pool2')

        conv3 = tf.layers.conv3d(inputs=pool2,
                                          filters=7,
                                          kernel_size=[1, 5, 5],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          data_format='channels_last',
                                          name='conv3')

        pool3 = tf.layers.max_pooling3d(inputs=conv3, 
                                                pool_size=[1, 2, 2], 
                                                strides=[1, 2, 2], 
                                                data_format='channels_last',
                                                name='pool3')


        conv4 = tf.layers.conv3d(inputs=pool3,
                                          filters=7,
                                          kernel_size=[1, 5, 5],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          data_format='channels_last',
                                          name='conv4')

        pool4 = tf.layers.max_pooling3d(inputs=conv4, 
                                                pool_size=[1, 2, 2], 
                                                strides=[1, 2, 2], 
                                                data_format='channels_last',
                                                name='pool4')
        conv5 = tf.layers.conv3d(inputs=pool4,
                                          filters=10,
                                          kernel_size=[1, 5, 5],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          data_format='channels_last',
                                          name='conv5')

        pool5 = tf.layers.max_pooling3d(inputs=conv5, 
                                                pool_size=[1, 2, 2], 
                                                strides=[1, 2, 2], 
                                                data_format='channels_last',
                                                name='pool5')

        conv6 = tf.layers.conv3d(inputs=pool5,
                                          filters=10,
                                          kernel_size=[1, 5, 5],
                                          padding='valid',
                                          activation=tf.nn.relu,
                                          data_format='channels_last',
                                          name='conv6')

        pool6 = tf.layers.max_pooling3d(inputs=conv6, 
                                                pool_size=[1, 2, 2], 
                                                strides=[1, 2, 2], 
                                                data_format='channels_last',
                                                name='pool6')

        feature_dim = pool6.shape[1] * pool6.shape[2] * pool6.shape[3] * pool6.shape[4]
        pool6 = tf.reshape(pool6, [-1, feature_dim])
        
        fc = tf.layers.dense(pool6, 30, activation=tf.nn.relu, name='fc')
        dropout = tf.layers.dropout(fc, 
                                    self.keep_prob,  
                                    training=self.training, 
                                    name='dropout')

        self.logits = tf.layers.dense(dropout, self.n_classes, name='logits')


    def loss(self):
        '''
        define loss function
        use softmax cross entropy with logits as the loss function
        compute mean cross entropy, softmax is applied internally
        '''
        # 
        with tf.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
            self.loss = tf.reduce_mean(entropy, name='loss')
            # self.loss = tf.losses.mean_squared_error(self.label, self.logits)
            
    
    def optimize(self):
        '''
        Define training op
        using Adam Gradient Descent to minimize cost
        '''
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.gstep)

    def summary(self):
        '''
        Create summaries to write on TensorBoard
        '''
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.summary.merge_all()
    
    def eval(self):
        '''
        Count the number of right predictions in a batch
        '''
        with tf.name_scope('predict'):
            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    def build(self):
        '''
        Build the computation graph
        '''
        self.get_data()
        self.inference()
        self.loss()
        self.optimize()
        self.eval()
        self.summary()

    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init) 
        self.training = True
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l, summaries = sess.run([self.opt, self.loss, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: {1}'.format(step, l))
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        if epoch % 2 == 0:
            saver.save(sess, 'checkpoints/convnet_layers/cnn_to_act_single_img', step)
        
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss/n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))
        return step

    def eval_once(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = False
        total_correct_preds = 0
        
        try:
            while True:
                accuracy_batch, summaries = sess.run([self.accuracy, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += accuracy_batch
        except tf.errors.OutOfRangeError:
            pass

        print('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds/self.n_test))
        print('Took: {0} seconds'.format(time.time() - start_time))

    def train(self, n_epochs):
        '''
        The train function alternates between training one epoch and evaluating
        '''
        utils.safe_mkdir('checkpoints')
        utils.safe_mkdir('checkpoints/convnet_layers')
        writer = tf.summary.FileWriter('./graphs/convnet_layers', tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_layers/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            
            step = self.gstep.eval()

            for epoch in range(n_epochs):
                step = self.train_one_epoch(sess, saver, self.train_init, writer, epoch, step)
                self.eval_once(sess, self.test_init, writer, epoch, step)
        writer.close()

if __name__ == '__main__':
    model = ConvNet()
    model.build()
    model.train(n_epochs=5)