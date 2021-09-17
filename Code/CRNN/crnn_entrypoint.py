import numpy as np
import matplotlib.pyplot as plt
import os
import editdistance
import pickle
import time

from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint

from CRNN.crnn_model import CRNN, CRNNv2
from CRNN.crnn_data import InputGenerator
from CRNN.crnn_utils import decode
from SegLink.ssd_training import Logger, ModelSnapshot
from CRNN.crnn_utils import alphabet87 as alphabet

from SegLink.data_synthtext import GTUtility

def train():
	with open('gt_util_synthtext_seglink.pkl', 'rb') as f:
		gt_util = pickle.load(f)
	gt_util_train, gt_util_other = gt_util.split(gt_util, split=0.95)
	gt_util_val, gt_util_test = gt_util.split(gt_util_other, split=0.2)

	input_width = 256
	input_height = 32
	batch_size = 128
	input_shape = (input_width, input_height, 1)

	model, model_pred = CRNNv2(input_shape, len(alphabet), gru=False)
	model.load_weights('./checkpoints/weights.200000.h5', by_name=True)

	for layer in model.layers:
		layer.trainable = False
	
	unfreeze = ['conv4_11111', 
				'batchnorm111', 
				'act111', 
				'conv4_22222', 
				'batchnorm222', 
				'act222',
				'conv5_11111', 
				'batchnorm333',
				'act333',
				'conv5_22222',
				'batchnorm444',
				'act444'
			   ]
			   
	for layer in model.layers:
		if layer.name in unfreeze:
			layer.trainable = True

	max_string_len = model_pred.output_shape[1]

	gen_train = InputGenerator(gt_util_train, batch_size, alphabet, input_shape[:2], grayscale=True, max_string_len=max_string_len)
	gen_val = InputGenerator(gt_util_val, batch_size, alphabet, input_shape[:2], grayscale=True, max_string_len=max_string_len)

	checkdir = './checkpoints/' + time.strftime('%Y%m%d%H%M') + '_' + 'crnn_fine_tune'
	if not os.path.exists(checkdir):
		os.makedirs(checkdir)

	# with open(checkdir+'/source.py','wb') as f:
	# 	source = ''.join(['# In[%i]\n%s\n\n' % (i, In[i]) for i in range(len(In))])
	# 	f.write(source.encode())

	optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
	#optimizer = Adam(lr=0.02, epsilon=0.001, clipnorm=1.)

	model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

	model.fit_generator(generator=gen_train.generate(), 
						steps_per_epoch=gt_util_train.num_objects // batch_size,
						epochs=100,
						validation_data=gen_val.generate(),
						validation_steps=gt_util_val.num_objects // batch_size,
						callbacks=[ModelCheckpoint(checkdir+'/weights.{epoch:03d}.h5', verbose=1, save_weights_only=True), ModelSnapshot(checkdir, 10000), Logger(checkdir)],
						initial_epoch=0)

if __name__ == "__main__":
	train()























