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
import cv2

from SegLink.sl_model import SL512, SL512v2, SL512_trunc, SL512v2_trunc
from SegLink.sl_utils import PriorUtil
from SegLink.ssd_utils import load_weights, calc_memory_usage
from SegLink.ssd_training import Logger, LearningRateDecay
from SegLink.sl_training import SegLinkLoss
from SegLink.ssd_model import ssd512_body
from SegLink.ssd_data import InputGenerator, preprocess

from numpy import linalg
from scipy import sparse

import scipy.optimize as optimize

def get_ABC():
	a_model, minimodel = SL512_trunc()
	prior_util = PriorUtil(a_model)
	image_size = a_model.image_size
	weights_path = "./SegLink/checkpoints/original_sl_synth/weights.001.h5"
	load_weights(minimodel, weights_path)
	b_model, minimodel2 = SL512v2_trunc()
	load_weights(minimodel2, weights_path)

	for layer in minimodel.layers:
	    if layer.name == "conv5_3":
	        original_weights = layer.get_weights()
	for layer in minimodel2.layers:
	    if layer.name == "conv5_3_1":
	        dw_weights = layer.get_weights()
	    elif layer.name == "conv5_3_2":
	        pw_weights = layer.get_weights()
        
	return np.array(original_weights)[0], np.array(dw_weights)[0], np.array(pw_weights)[0]

if __name__ == "__main__":
	
	# objective function
	def f(params, A):
		total = 0
		B, C = params[:3*3*512*1], params[3*3*512*1:]
		for k in range(0,512):
			for m in range(0,512):
				for n in range(0,9):
					total += (A[k*m*n] - B[m*n] * C[k*m])**2
		print(total)
		return total

	# get matrices
	A, B, C = get_ABC()

	# flatten into 1d vectors
	A = A.ravel()
	B = B.ravel()
	C = C.ravel()

	# perform minimization
	x = np.hstack([B.flatten(), C.flatten()])
	result = optimize.minimize(f, x, args=(A), method='BFGS', tol=1e-3, options={'eps': 0.1})

	# print result
	if result.success:
	    fitted_params = result.x
	    print(fitted_params)
	else:
	    raise ValueError(result.message)

















