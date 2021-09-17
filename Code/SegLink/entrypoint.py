# IMPORT LIBRARIES
import keras
from keras.callbacks import Callback
import tensorflow as tf
import os
import time
import pickle
from keras.utils import multi_gpu_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random

from SegLink.sl_model import SL512, SL512v2
from SegLink.ssd_data import InputGenerator
from SegLink.sl_utils import PriorUtil
from SegLink.ssd_utils import load_weights, calc_memory_usage
from SegLink.ssd_training import Logger, LearningRateDecay
from SegLink.sl_training import SegLinkLoss
from SegLink.ssd_model import ssd512_body

def train():

	# IMPORT SYNTEXT DATA
	random.seed(1337)
	print("[i]: Importing Data")
	from data_synthtext import GTUtility
	with open('gt_util_synthtext_seglink.pkl', 'rb') as f:
		gt_util = pickle.load(f)
	gt_util_train, gt_util_other = gt_util.split(gt_util, split=0.1)
	gt_util_val, foo = gt_util.split(gt_util_other, split=0.01)
	gt_util_test, bar = gt_util.split(foo, split=0.01)
	# from data_svt import GTUtility
	# gt_util_training1 = GTUtility('data/SVT/', polygon=True)
	# gt_util_test1 = GTUtility('data/SVT/', test=True, polygon=True)
	# gt_util_training1, gt_util_val_1 = gt_util_training1.split(gt_util_training1, split=0.8)
	# print("[i] StreetView Text Training:", gt_util_training1)
	# print("[i] StreetView Text Validation:", gt_util_val_1)
	# print("[i] StreetView Text Testing:", gt_util_test1)

	# from data_icdar2015fst import GTUtility
	# gt_util_train = GTUtility('data/ICDAR2015_FST/', polygon=True)
	# gt_util_test = GTUtility('data/ICDAR2015_FST/', test=True, polygon=False)
	# gt_util_train, gt_util_val = gt_util_train.split(gt_util_train, split=0.8)
	# print("[i] Icdar2015 Training:", gt_util_train)
	# print("[i] Icdar2015 Validation:", gt_util_val)
	# print("[i] Icdar2015 Testing:", gt_util_test)

	# from data_icdar2015ist import GTUtility
	# gt_util_train_2015 = GTUtility('data/ICDAR2015_IST/', test=False)
	# gt_util_test_2015 = GTUtility('data/ICDAR2015_IST/', test=True)

	# from data_icdar2015fst import GTUtility
	# gt_util_train_2013 = GTUtility('data/ICDAR2015_FST/', test=False, polygon=True)
	# gt_util_test_2013 = GTUtility('data/ICDAR2015_FST/', test=True, polygon=True)

	# from data_svt import GTUtility
	# gt_util_train_svt = GTUtility('data/SVT/', test=False)
	# gt_util_test_svt = GTUtility('data/SVT/', test=True)

	# gt_util_train = gt_util_train_2015
	# gt_util_val = gt_util_test_2013
		
		# INITIALIZE MODEL
	print("[i]: Initializing SegLink Model")
	# with tf.device("/cpu:0"):
	# model = SL512()
	model = SL512v2()
	prior_util = PriorUtil(model)
	image_size = model.image_size
	weights_path = "./checkpoints/weights.001.h5"
	layer_list = ['conv1_1', 'conv1_2',
				  'conv2_1', 'conv2_2',
				  'conv3_1', 'conv3_2', 'conv3_3']
	# 			  'conv5_1', 'conv5_2', 'conv5_3',
	# 			  'fc6', 'fc7',
	# 			  'conv6_1', 'conv6_2',
	# 			  'conv7_1', 'conv7_2',
	# 			  'conv8_1', 'conv8_2',
	# 			  'conv9_1', 'conv9_2',
	# 			  ]
	# load_weights(model, weights_path, layer_list)

	if weights_path is not None:
		load_weights(model, weights_path)
	# model.load_weights(weights_path, by_name=True)

	# freeze = ['conv1_1', 'conv1_2',
 #          'conv2_1', 'conv2_2',
 #          'conv3_1', 'conv3_2', 'conv3_3'
 #          ]

	dscs = ['conv5_3_1', 'conv5_3_2']
	# unfreeze = []

	# freeze = ['conv1_1', 
	# 			'conv1_2', 
	# 			'pool1', 
	# 			'conv2_1', 
	# 			'conv2_2', 
	# 			'pool2',
	# 			'conv5_11111', 
	# 			'conv3_1',
	# 			'conv3_2',
	# 			'conv3_3',
	# 			'pool3'
	# 		   ]

	# for layer in model.layers:
	#     layer.trainable = True

	for layer in model.layers:
		if layer.name in layer_list:
			layer.trainable = False
	   
	# for layer in model.layers:
	# 	if layer.name in freeze:
	# 		layer.trainable = False

	print("++++++++++++++++++++")
	for layer in model.layers:
		print(layer.name, layer.trainable)
	print("++++++++++++++++++++")

	# DEFINE GENERATOR
	# gen = InputGenerator(
	# 		gt_util_train=gt_util_train, 
	# 		gt_util_val=gt_util_val, 
	# 		prior_util=prior_util, 
	# 		batch_size=8, 
	# 		input_size=image_size, 
	# 		saturation_var=0.5,
	# 		brightness_var=0.5,
	# 		contrast_var=0.5,
	# 		lighting_std=0.5,
	# 		hflip_prob=0.0,
	# 		vflip_prob=0.0,
	# 		do_crop=True,
	# 		add_noise=True,
	# 		crop_area_range=[0.1, 1.0],
	# 		aspect_ratio_range=[3/4, 4/3])
	batch_size=16
	gen_train = InputGenerator(gt_util_train, prior_util, batch_size, model.image_size, augmentation=True)
	gen_val = InputGenerator(gt_util_val, prior_util, batch_size, model.image_size, augmentation=True)

	# SET EPOCH FLAGS
	epochs = 1000
	initial_epoch = 0

	# CREATE CHECKPOINT PATH
	print("[i]: Making Savepath")
	checkdir = './checkpoints/' + time.strftime('%Y%m%d%H%M') + '_' + 'lasttry'
	if not os.path.exists(checkdir):
		os.makedirs(checkdir)

	# COMPILE MODEL
	print("[i]: Compiling Model")
	for layer in model.layers:
		if layer.name in dscs:
			layer.kernel_regularizer = keras.regularizers.l1(1e-2)
		
	optim = keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=0.001, decay=0.0)
	loss = SegLinkLoss(lambda_offsets=1.0, lambda_links=1.0, neg_pos_ratio=3.0)
	model.compile(optimizer=optim, loss=loss.compute, metrics=loss.metrics)
	callbacks = [keras.callbacks.ModelCheckpoint(checkdir+'/weights.{epoch:03d}.h5', verbose=1, save_weights_only=True), Logger(checkdir)]

	# TRAIN MODEL
	print("[i]: Beginging Training Process")
	print("\n")

	H = model.fit_generator( 
			generator=gen_train.generate(),
			steps_per_epoch=gen_train.num_batches, 
			epochs=epochs, 
			verbose=1, 
			callbacks=callbacks, 
			validation_data=gen_val.generate(), 
			validation_steps=gen_val.num_batches, 
			class_weight=None,
			max_queue_size=1,  
			initial_epoch=initial_epoch)

if __name__ == "__main__":
	train()





