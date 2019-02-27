from __future__ import division
from __future__ import print_function
import os
import time
from glob import glob
import tensorflow as tf
from collections import namedtuple

from module import *
from utils import *

import uvc
import cv2
import numpy as np
from functools import partial
from time import time,sleep
import argparse

import asyncio
import logging
import base64

try:
    from multiprocessing import Process, forking_enable, freeze_support
except ImportError:
    try:
        # python3
        from multiprocessing import Process, Pool, Manager, Queue, set_start_method, freeze_support

        def forking_enable(_):
            set_start_method('spawn')
    except ImportError:
        # python2 macos
        from billiard import Process, forking_enable, freeze_support

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='large_dataset/Aakash', help='path of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=10, help='# of epoch')
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=200, help='# of epoch to decay lr')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=1000, help='save a model every save_freq iterations')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=100, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='checkpoint/aakash_400', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0, help='weight on L1 term in objective')
parser.add_argument('--use_resnet', dest='use_resnet', type=bool, default=True, help='generation network using reidule block')
parser.add_argument('--use_lsgan', dest='use_lsgan', type=bool, default=True, help='gan loss defined in lsgan')
parser.add_argument('--max_size', dest='max_size', type=int, default=50, help='max size of image pool, 0 means do not use image pool')

""" set variable initialize tensorflow variables """
init_op = tf.global_variables_initializer()

""" setup TCP IP ADDRESS and PORT """
TCP_IP = '10.1.7.20'
TCP_PORT1 = 5001

""" Set the wait time for storing data in queue (default minimum is 0.2 sec) """
wait = 0.2

""" 
this is for the pupil cameras' hardware access configuration
-> FORMAT = 'HOLOLENS' (for the cameras' attached to the HoloLens)
-> FORMAT = 'STANDALONE' (for the cameras' attached to the standalone head mount)
(this is subject to change for each hardware used)
"""
FORMAT = 'HOLOLENS'

""" Set the number of pupil cameras for each person """
TOTAL_CAMERAS_CONNECTED = 3

""" get the device list of different pupil cameras (as per PyUVC for Universal Video Device Class (UVC) protocol) """
dev_list = uvc.device_list()

""" set up asynchronous TCP/IP communication """
async def tcp_echo_client(data, loop, TCP_IP, TCP_PORT):
    reader, writer = await asyncio.open_connection(TCP_IP, TCP_PORT, loop=loop)
    writer.write(str(len(data)).encode() + str(data).encode())

""" Access the right eye camera """
class PupilCamReye():
    def __init__(self, queue, config1):
        self.sess = tf.Session(config=config1)
        self.sess.run(init_op)

        self.image_cycgans = np.zeros((512, 288, 3), 'uint8')
        self.image_tensor = tf.placeholder(tf.uint8, [512, 288, 3])

        self.leye_index = 0
        self.reye_index = 0
        self.mouth_index = 0
        self.count = 0
        self.queue = queue
        self.format = FORMAT

        self.start()
        self.my_name(self.reye_index, 'RIGHT EYE CAPTURE', (640, 480, 60), 'bgr', 1.3)

    def start(self):

        """ get camera indices from the HoloLens integrated cameras """
        if self.format == 'HOLOLENS':
            for i in range(len(dev_list)):
                if dev_list[i]['manufacturer'] == 'Pupil Cam1 ID0':
                    if self.reye_index == 0:
                        self.count = self.count + 1
                        self.reye_index = i
                        print('device_list index: ' + str(i)
                              + ' --> Right eye camera found with manufacturer id: '
                              + dev_list[i]['manufacturer'])
                    else:
                        self.count = self.count + 1
                        self.mouth_index = i
                        print('device_list index: ' + str(i)
                              + ' --> Mouth camera found with manufacturer id: '
                              + dev_list[i]['manufacturer'])

                elif dev_list[i]['manufacturer'] == 'Pupil Cam1 ID1':
                    self.count = self.count + 1
                    self.leye_index = i
                    print('device_list index: ' + str(i)
                          + ' --> Left eye camera found with manufacturer id: '
                          + dev_list[i]['manufacturer'])
                else:
                    print('device_list index: ' + str(i)
                          + ' --> No camera found, manufacturer id: '
                          + dev_list[i]['manufacturer'])

            """ if all the three cameras are not found then exit the program """
            print('\n')
            if self.count != TOTAL_CAMERAS_CONNECTED:
                print('All the cameras are not connected...')
                print('Check connection and retry running !!!')
                exit(0)

            print(self.reye_index)
            print(self.mouth_index)
            print(self.leye_index)

        """ get camera indices from the standalone frame integrated cameras """
        if self.format == 'STANDALONE':
            for i in range(len(dev_list)):
                if dev_list[i]['manufacturer'] == 'Pupil Cam1 ID1':
                    if self.mouth_index == 0:
                        self.count = self.count + 1
                        self.mouth_index = i
                        print('device_list index: ' + str(i)
                              + ' --> Mouth camera found with manufacturer id: '
                              + dev_list[i]['manufacturer'])
                    else:
                        self.count = self.count + 1
                        self.leye_index = i
                        print('device_list index: ' + str(i)
                              + ' --> Left eye camera found with manufacturer id: '
                              + dev_list[i]['manufacturer'])

                elif dev_list[i]['manufacturer'] == 'Pupil Cam1 ID0':
                    self.count = self.count + 1
                    self.reye_index = i
                    print('device_list index: ' + str(i)
                          + ' --> Right eye camera found with manufacturer id: '
                          + dev_list[i]['manufacturer'])
                else:
                    print('device_list index: ' + str(i)
                          + ' --> No camera found, manufacturer id: '
                          + dev_list[i]['manufacturer'])

            """ if all the three cameras are not found then exit the program """
            print('\n')
            if self.count != TOTAL_CAMERAS_CONNECTED:
                print('All the cameras are not connected...')
                print('Check connection and retry running !!!')
                exit(0)

            print(self.reye_index)
            print(self.mouth_index)
            print(self.leye_index)

    """ Function to capture frames from 3 cameras using multiprocessing """
    def my_name(self, index, title, mode=(640, 480, 60), format='bgr', bandwidth_factor=1.3):
        leye_x = 153
        leye_y = 180

        reye_x = 25
        reye_y = 180

        eye_width = 107
        eye_height = 80

        cap = uvc.Capture(dev_list[index]['uid'])
        cap.bandwidth_factor = bandwidth_factor
        cap.frame_mode = mode

        while True:
            frame = cap.get_frame_robust()

            if format == 'bgr':
                data = frame.img
            else:
                data = frame.img

            if title == 'RIGHT EYE CAPTURE':
                self.image_cycgans.fill(0)
                imgr = cv2.flip(data, -1)  # vertical

                # compute homography right eye
                pts_srcr = np.array([[0, 0], [imgr.shape[1], 0], [0, imgr.shape[0]], [imgr.shape[1], imgr.shape[0]]])
                pts_dstr = np.array([[reye_x, reye_y], [reye_x + eye_width, reye_y], [reye_x, reye_y + eye_height],
                                     [reye_x + eye_width, reye_y + eye_height]])
                h, status = cv2.findHomography(pts_srcr, pts_dstr)
                new_image22 = cv2.warpPerspective(imgr, h, (self.image_cycgans.shape[1], self.image_cycgans.shape[0]))
                self.image_cycgans = self.image_cycgans + new_image22
                image_temp = self.sess.run(self.image_tensor, feed_dict={self.image_tensor: self.image_cycgans})
                self.queue.put(image_temp)
                sleep(wait)

        cap = None


""" Access the left eye camera """
class PupilCamLeye():
    def __init__(self, queue, config1):
        self.sess = tf.Session(config=config1)
        self.sess.run(init_op)

        self.image_cycgans = np.zeros((512, 288, 3), 'uint8')
        self.image_tensor = tf.placeholder(tf.uint8, [512, 288, 3])

        self.leye_index = 0
        self.reye_index = 0
        self.mouth_index = 0
        self.count = 0
        self.queue = queue
        self.format = FORMAT

        self.start()
        self.my_name(self.leye_index, 'LEFT EYE CAPTURE', (640, 480, 60), 'bgr', 1.3)

    def start(self):

        """ get camera indices from the HoloLens integrated cameras """
        if self.format == 'HOLOLENS':
            for i in range(len(dev_list)):
                if dev_list[i]['manufacturer'] == 'Pupil Cam1 ID0':
                    if self.reye_index == 0:
                        self.count = self.count + 1
                        self.reye_index = i
                        print('device_list index: ' + str(i)
                              + ' --> Right eye camera found with manufacturer id: '
                              + dev_list[i]['manufacturer'])
                    else:
                        self.count = self.count + 1
                        self.mouth_index = i
                        print('device_list index: ' + str(i)
                              + ' --> Mouth camera found with manufacturer id: '
                              + dev_list[i]['manufacturer'])

                elif dev_list[i]['manufacturer'] == 'Pupil Cam1 ID1':
                    self.count = self.count + 1
                    self.leye_index = i
                    print('device_list index: ' + str(i)
                          + ' --> Left eye camera found with manufacturer id: '
                          + dev_list[i]['manufacturer'])
                else:
                    print('device_list index: ' + str(i)
                          + ' --> No camera found, manufacturer id: '
                          + dev_list[i]['manufacturer'])

            """ if all the three cameras are not found then exit the program """
            print('\n')
            if self.count != TOTAL_CAMERAS_CONNECTED:
                print('All the cameras are not connected...')
                print('Check connection and retry running !!!')
                exit(0)

            print(self.reye_index)
            print(self.mouth_index)
            print(self.leye_index)

        """ get camera indices from the standalone frame integrated cameras """
        if self.format == 'STANDALONE':
            for i in range(len(dev_list)):
                if dev_list[i]['manufacturer'] == 'Pupil Cam1 ID1':
                    if self.mouth_index == 0:
                        self.count = self.count + 1
                        self.mouth_index = i
                        print('device_list index: ' + str(i)
                              + ' --> Mouth camera found with manufacturer id: '
                              + dev_list[i]['manufacturer'])
                    else:
                        self.count = self.count + 1
                        self.leye_index = i
                        print('device_list index: ' + str(i)
                              + ' --> Left eye camera found with manufacturer id: '
                              + dev_list[i]['manufacturer'])

                elif dev_list[i]['manufacturer'] == 'Pupil Cam1 ID0':
                    self.count = self.count + 1
                    self.reye_index = i
                    print('device_list index: ' + str(i)
                          + ' --> Right eye camera found with manufacturer id: '
                          + dev_list[i]['manufacturer'])
                else:
                    print('device_list index: ' + str(i)
                          + ' --> No camera found, manufacturer id: '
                          + dev_list[i]['manufacturer'])

            """ if all the three cameras are not found then exit the program """
            print('\n')
            if self.count != TOTAL_CAMERAS_CONNECTED:
                print('All the cameras are not connected...')
                print('Check connection and retry running !!!')
                exit(0)

            print(self.reye_index)
            print(self.mouth_index)
            print(self.leye_index)

    """ Function to capture frames from 3 cameras using multiprocessing """
    def my_name(self, index, title, mode=(640, 480, 60), format='bgr', bandwidth_factor=1.3):
        leye_x = 153
        leye_y = 180

        eye_width = 107
        eye_height = 80

        cap = uvc.Capture(dev_list[index]['uid'])
        cap.bandwidth_factor = bandwidth_factor
        cap.frame_mode = mode

        while True:
            frame = cap.get_frame_robust()

            if format == 'bgr':
                data = frame.img
            else:
                data = frame.img

            if title == 'LEFT EYE CAPTURE':
                self.image_cycgans.fill(0)
                imgl = data  # cv2.flip(data, 1)  # vertical

                # compute homography left eye
                pts_srcl = np.array([[0, 0], [imgl.shape[1], 0], [0, imgl.shape[0]], [imgl.shape[1], imgl.shape[0]]])
                pts_dstl = np.array([[leye_x, leye_y], [leye_x + eye_width, leye_y], [leye_x, leye_y + eye_height],
                                     [leye_x + eye_width, leye_y + eye_height]])
                h, status = cv2.findHomography(pts_srcl, pts_dstl)
                new_image11 = cv2.warpPerspective(imgl, h, (self.image_cycgans.shape[1], self.image_cycgans.shape[0]))
                self.image_cycgans = self.image_cycgans + new_image11
                image_temp = self.sess.run(self.image_tensor, feed_dict={self.image_tensor: self.image_cycgans})
                self.queue.put(image_temp)
                sleep(wait)

        cap = None


""" Access the mouth camera """
class PupilCamMouth():
    def __init__(self, queue, config1):
        self.sess = tf.Session(config=config1)
        self.sess.run(init_op)
        self.image_cycgans = np.zeros((512, 288, 3), 'uint8')
        self.image_tensor = tf.placeholder(tf.uint8, [512, 288, 3])

        self.leye_index = 0
        self.reye_index = 0
        self.mouth_index = 0
        self.count = 0
        self.queue = queue
        self.format = FORMAT

        self.start()
        self.my_name(self.mouth_index, 'MOUTH CAPTURE', (640, 480, 60), 'bgr', 1.3)

    def start(self):

        """ get camera indices from the HoloLens integrated cameras """
        if self.format == 'HOLOLENS':
            for i in range(len(dev_list)):
                if dev_list[i]['manufacturer'] == 'Pupil Cam1 ID0':
                    if self.reye_index == 0:
                        self.count = self.count + 1
                        self.reye_index = i
                        print('device_list index: ' + str(i)
                              + ' --> Right eye camera found with manufacturer id: '
                              + dev_list[i]['manufacturer'])
                    else:
                        self.count = self.count + 1
                        self.mouth_index = i
                        print('device_list index: ' + str(i)
                              + ' --> Mouth camera found with manufacturer id: '
                              + dev_list[i]['manufacturer'])

                elif dev_list[i]['manufacturer'] == 'Pupil Cam1 ID1':
                    self.count = self.count + 1
                    self.leye_index = i
                    print('device_list index: ' + str(i)
                          + ' --> Left eye camera found with manufacturer id: '
                          + dev_list[i]['manufacturer'])
                else:
                    print('device_list index: ' + str(i)
                          + ' --> No camera found, manufacturer id: '
                          + dev_list[i]['manufacturer'])

            """ if all the three cameras are not found then exit the program """
            print('\n')
            if self.count != TOTAL_CAMERAS_CONNECTED:
                print('All the cameras are not connected...')
                print('Check connection and retry running !!!')
                exit(0)

            print(self.reye_index)
            print(self.mouth_index)
            print(self.leye_index)

        """ get camera indices from the standalone frame integrated cameras """
        if self.format == 'STANDALONE':
            for i in range(len(dev_list)):
                if dev_list[i]['manufacturer'] == 'Pupil Cam1 ID1':
                    if self.mouth_index == 0:
                        self.count = self.count + 1
                        self.mouth_index = i
                        print('device_list index: ' + str(i)
                              + ' --> Mouth camera found with manufacturer id: '
                              + dev_list[i]['manufacturer'])
                    else:
                        self.count = self.count + 1
                        self.leye_index = i
                        print('device_list index: ' + str(i)
                              + ' --> Left eye camera found with manufacturer id: '
                              + dev_list[i]['manufacturer'])

                elif dev_list[i]['manufacturer'] == 'Pupil Cam1 ID0':
                    self.count = self.count + 1
                    self.reye_index = i
                    print('device_list index: ' + str(i)
                          + ' --> Right eye camera found with manufacturer id: '
                          + dev_list[i]['manufacturer'])
                else:
                    print('device_list index: ' + str(i)
                          + ' --> No camera found, manufacturer id: '
                          + dev_list[i]['manufacturer'])

            """ if all the three cameras are not found then exit the program """
            print('\n')
            if self.count != TOTAL_CAMERAS_CONNECTED:
                print('All the cameras are not connected...')
                print('Check connection and retry running !!!')
                exit(0)

            print(self.reye_index)
            print(self.mouth_index)
            print(self.leye_index)

    """ Function to capture frames from 3 cameras using multiprocessing """
    def my_name(self, index, title, mode=(640, 480, 60), format='bgr', bandwidth_factor=1.3):
        mouth_x = 1
        mouth_y = 267

        mouth_width = 284
        mouth_height = 190

        cap = uvc.Capture(dev_list[index]['uid'])
        cap.bandwidth_factor = bandwidth_factor
        cap.frame_mode = mode

        while True:
            frame = cap.get_frame_robust()

            if format == 'bgr':
                data = frame.img
            else:
                data = frame.img

            if title == 'MOUTH CAPTURE':
                self.image_cycgans.fill(0)
                imgm = cv2.flip(data, 0)  # horizontal

                # compute homography mouth
                pts_srcm = np.array([[0, 0], [imgm.shape[1], 0], [0, imgm.shape[0]], [imgm.shape[1], imgm.shape[0]]])
                pts_dstm = np.array([[mouth_x, mouth_y], [mouth_x + mouth_width, mouth_y], [mouth_x, mouth_y + mouth_height],
                                     [mouth_x + mouth_width, mouth_y + mouth_height]])
                h, status = cv2.findHomography(pts_srcm, pts_dstm)
                new_image33 = cv2.warpPerspective(imgm, h, (self.image_cycgans.shape[1], self.image_cycgans.shape[0]))
                self.image_cycgans = self.image_cycgans + new_image33
                image_temp = self.sess.run(self.image_tensor, feed_dict={self.image_tensor: self.image_cycgans})
                self.queue.put(image_temp)
                sleep(wait)

        cap = None


""" CycleGan trained network, checkpoint, etc. to test data realtime """
class cyclegan(object):
    def __init__(self, queue, config1, args):
        self.sess = tf.Session(config=config1)
        self.sess.run(init_op)

        self.batch_size = args.batch_size
        self.image_size = args.fine_size
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.L1_lambda = args.L1_lambda
        self.dataset_dir = args.dataset_dir

        self.imager = tf.placeholder(tf.uint8, [512, 288, 3])
        self.imagel = tf.placeholder(tf.uint8, [512, 288, 3])
        self.imagem = tf.placeholder(tf.uint8, [512, 288, 3])

        self.queue = queue

        self.discriminator = discriminator
        if args.use_resnet:
            self.generator = generator_resnet
        else:
            self.generator = generator_unet
        if args.use_lsgan:
            self.criterionGAN = mae_criterion
        else:
            self.criterionGAN = sce_criterion

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                              gf_dim df_dim output_c_dim is_training')
        self.options = OPTIONS._make((args.batch_size, args.fine_size,
                                      args.ngf, args.ndf, args.output_nc,
                                      args.phase == 'train'))

        self._build_model()
        self.saver = tf.train.Saver()
        self.pool = ImagePool(args.max_size)
        self.test(args)

    def _build_model(self):
        self.real_data = tf.placeholder(tf.float32,
                                        [None, self.image_size, self.image_size,
                                         self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_images')

        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        self.fake_B = self.generator(self.real_A, self.options, False, name="generatorA2B")
        self.fake_A_ = self.generator(self.fake_B, self.options, False, name="generatorB2A")
        self.fake_A = self.generator(self.real_B, self.options, True, name="generatorB2A")
        self.fake_B_ = self.generator(self.fake_A, self.options, True, name="generatorA2B")

        self.DB_fake = self.discriminator(self.fake_B, self.options, reuse=False, name="discriminatorB")
        self.DA_fake = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorA")
        self.g_loss_a2b = self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
        self.g_loss_b2a = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
        self.g_loss = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
            + self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)

        self.fake_A_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.input_c_dim], name='fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.output_c_dim], name='fake_B_sample')
        self.DB_real = self.discriminator(self.real_B, self.options, reuse=True, name="discriminatorB")
        self.DA_real = self.discriminator(self.real_A, self.options, reuse=True, name="discriminatorA")
        self.DB_fake_sample = self.discriminator(self.fake_B_sample, self.options, reuse=True, name="discriminatorB")
        self.DA_fake_sample = self.discriminator(self.fake_A_sample, self.options, reuse=True, name="discriminatorA")

        self.db_loss_real = self.criterionGAN(self.DB_real, tf.ones_like(self.DB_real))
        self.db_loss_fake = self.criterionGAN(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample))
        self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2
        self.da_loss_real = self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real))
        self.da_loss_fake = self.criterionGAN(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
        self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2
        self.d_loss = self.da_loss + self.db_loss

        self.g_loss_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b)
        self.g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.g_sum = tf.summary.merge([self.g_loss_a2b_sum, self.g_loss_b2a_sum, self.g_loss_sum])
        self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
        self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real)
        self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake)
        self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
        self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)
        self.d_sum = tf.summary.merge(
            [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
             self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
             self.d_loss_sum]
        )

        self.test_A = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.input_c_dim], name='test_A')
        self.test_B = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.output_c_dim], name='test_B')
        self.testB = self.generator(self.test_A, self.options, True, name="generatorA2B")
        self.testA = self.generator(self.test_B, self.options, True, name="generatorB2A")

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        for var in t_vars: print(var.name)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test(self, args):
        """Test cyclegan"""
        # init_op = tf.global_variables_initializer()
        # self.sess.run(init_op)

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        out_var, in_var = (self.testB, self.test_A)

        sample_image = np.zeros((512, 288, 3), 'uint8')

        while True:
            sample_image.fill(0)

            r = self.queue.get()
            r = self.sess.run(self.imager, {self.imager: r})

            l = self.queue.get()
            l = self.sess.run(self.imagel, {self.imagel: l})

            m = self.queue.get()
            m = self.sess.run(self.imagem, {self.imagem: m})

            sample_image = r + l + m

            dispIR = cv2.resize(sample_image, (288, 512))
            sample_image = cv2.resize(sample_image, (256, 256))

            cv2.imshow('input', dispIR)
            cv2.moveWindow('input', 220, 30)
            cv2.waitKey(1)

            sample_image = sample_image.astype(np.float32)
            sample_image = sample_image / 127.5 - 1.
            sample_image1 = [sample_image]
            sample_image1 = np.array(sample_image1).astype(np.float32)
            fake_img = self.sess.run(out_var, feed_dict={in_var: sample_image1})
            f = create_image(fake_img, [1, 1])  # fake_img.reshape(256, 256, 3)

            """ get the right RGB color for display """
            f = f[:, :, ::-1]
            f = 255 * f
            f = f.astype(np.uint8)
            f = cv2.resize(f, (160, 120))
            # print(f.shape)

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            result, imgencode = cv2.imencode('.jpg', f, encode_param)
            # data = base64.b64encode(imgencode)
            # loop = asyncio.get_event_loop()
            # loop.run_until_complete(tcp_echo_client(data, loop, TCP_IP, TCP_PORT1))
            decimg = cv2.imdecode(np.array(imgencode), 1)

            dispRGB = cv2.resize(decimg, (288, 512))
            cv2.imshow('output', dispRGB)
            cv2.moveWindow('output', 288*2, 30)
            cv2.waitKey(10)


""" this queue is a multiprocessing queue to store frames (share between threads) """
q = Queue()

""" the main function that has all the process calls (thread safe) """
def main(_):
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    p1 = Process(target=PupilCamReye, args=(q, tfconfig))
    p1.start()
    p2 = Process(target=PupilCamLeye, args=(q, tfconfig))
    p2.start()
    p3 = Process(target=PupilCamMouth, args=(q, tfconfig))
    p3.start()

    args1 = parser.parse_args()

    proc = Process(target=cyclegan, args=(q, tfconfig, args1))
    proc.start()

""" start session from here """
if __name__ == '__main__':
    tf.app.run()
