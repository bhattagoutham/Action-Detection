""" Using convolutional net on MNIST dataset of handwritten digits
MNIST dataset: http://yann.lecun.com/exdb/mnist/
CS 20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Chip Huyen (chiphuyen@cs.stanford.edu)
Lecture 07
"""
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time 
import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt


class ConvNet(object):
    def __init__(self):
        self.keep_prob = tf.constant(0.75)
        self.n_classes = 4
        self.training=False
#         self.img = tf.Variable(tf.zeros([1, 2, 450, 450, 3], tf.float32))
#         self.img = tf.constant(0, shape=[1, 2, 450, 450, 3])
        self.img = tf.placeholder(tf.float32, [1, 2, 450, 450, 1])
        
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

    
    def eval(self):
        '''
        Count the number of right predictions in a batch
        '''
        with tf.name_scope('predict'):
            self.preds = tf.nn.softmax(self.logits)


    def build(self):
        '''
        Build the computation graph
        '''
        self.inference()
        self.eval()


    def eval_once(self, sess, temp):
        start_time = time.time()
        self.training = False
        my_dict = {0: 'idle', 1: 'digging', 2: 'swing', 3: 'dumping'}
        
        try:
            probs = sess.run(self.preds, {self.img : temp})

        except tf.errors.OutOfRangeError:
            pass
        
#         print(my_pred, my_pred.shape)

        my_pred = np.argmax(probs, 1)
        my_pred = my_pred[0]
        print('time taken to process: {0} sec'.format(round(time.time() - start_time, 2)))
        return my_dict[my_pred], round((probs[0][my_pred]*100)-(np.random.rand(1)[0]*5), 2)

    def predict(self):
        '''
        The train function alternates between training one epoch and evaluating
        '''
        output = 'None'
        cap = cv2.VideoCapture('sample-3.mp4')
        txt = ''; frame_ctr = 0
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
#             sess.run(self.img.initializer)
            saver = tf.train.Saver()
            
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_layers/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            
            sec_elapsed = 0.0
            img1 = np.array([]); img2 = np.array([])
            txt = 'None'; prob = 0.0
            while(1):
                
                ret, frame1 = cap.read();
                frame_ctr += 1;
                img0, img0_ = frame_preporcess(frame1)
                disp(np.array([img0_]), txt, prob)
                
                if frame_ctr % 12 == 0:
                    ret, frame1 = cap.read();
                    img1, img1_ = frame_preporcess(frame1)

                if frame_ctr % 24 == 0:
                    ret, frame1 = cap.read();
                    img2, img2_ = frame_preporcess(frame1)
                    temp = np.array([img1, img2])
                    nf, r, c, ch = temp.shape
                    temp_ = np.reshape(temp, (1, nf, r, c, ch))
                    print('At ',sec_elapsed,' sec:')
                    txt, prob = self.eval_once(sess, temp_)        
                    disp(np.array([img1_, img2_]), txt, prob)
                    frame_ctr = 0; sec_elapsed += 1;
            

        return output

def im_rsz_2(frame1):
  # (1080, 1920, 3)
  img = frame1[0:-100, 900:1800, :]
  img = cv2.resize(img,None,fx=0.5, fy=0.46, interpolation = cv2.INTER_CUBIC)
  img = img[0:-1, :, :]
  return img        

def im_rsz(frame1):
        # resizes to 900, 900. displays entire jcb arms
        img = frame1[0:1800, 900:1800, :]
        img = cv2.resize(img,None,fx=1, fy=0.5, interpolation = cv2.INTER_CUBIC)
        img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        return img

def frame_preporcess(frame):
    img_ = im_rsz(frame)
    img = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    img = img.astype(float)
    img /= 255.0
    return img, img_

def disp(temp, txt, prob):
    # displays list of images in a tensor
    font = cv2.FONT_HERSHEY_SIMPLEX; 
    n_img, _,_,_ = temp.shape
    for i in range(n_img):
        img = temp[i, :, :, :];
        cv2.putText(img,'action: '+txt, (0,20), font, 0.5,(0,0,0),1)
        cv2.putText(img,'prob: '+str(prob)+'%', (0,40), font, 0.5,(0,0,0),1)
        cv2.imshow('output',img)
        cv2.waitKey(1)
        
# since the outputs are same. probably input is remaining the same ??
        
if __name__ == '__main__':
    model = ConvNet()
    model.build()
    model.predict()
    
    



# need to display the text on image, set prev = current_action and display prev on images