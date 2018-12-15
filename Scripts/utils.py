import json
import numpy as np
import keras.backend as K
import tensorflow as tf

def precision(y_true, y_pred):
	y_pred_bin = K.cast(y_true > 0.5, K.floatx())
	y_true_inverse = K.cast(tf.math.logical_not(y_true > 0.5), K.floatx())
	y_pred_bin_inverse = K.cast(tf.math.logical_not(y_pred_bin))
	
	tp_1 = K.sum(y_true * y_pred_bin)
	tp_0 = K.sum(y_true_inverse * y_pred_bin_inverse)
	tp_sum = tf.stack([tp_0, tp_1])
	
	pred_sum_0 = K.sum(
		K.cast(K.equal(y_pred_bin, 0.), K.floatx())
	)
	pred_sum_1 = K.sum(
		K.cast(K.equal(y_pred_bin, 1.), K.floatx())
	)
	pred_sum = tf.stack([pred_sum_0, pred_sum_1])
	
	precision = tp_sum / pred_sum
	precision *= K.cast(tf.is_inf(precision), K.floatx())
	
def weighted_f1(y_true, y_pred):
	y_pred_bin = K.cast(y_true > 0.5, K.floatx())
	y_true_inverse = K.cast(tf.math.logical_not(y_true > 0.5), K.floatx())
	y_pred_bin_inverse = K.cast(tf.math.logical_not(y_pred_bin))
	
	tp_1 = K.sum(y_true * y_pred_bin)
	tp_0 = K.sum(y_true_inverse * y_pred_bin_inverse)
	
	tp_sum = K.concatenate([tp_0, tp_1])
	
	pred_sum_0 = K.sum(
		K.cast(K.equal(y_pred_bin, 0.), K.floatx())
	)
	pred_sum_1 = K.sum(
		K.cast(K.equal(y_pred_bin, 1.), K.floatx())
	)
	pred_sum = K.concatenate([pred_sum_0, pred_sum_1])
	
	true_sum_0 = K.sum(
		K.cast(K.equal(y_true_bin, 0.), K.floatx())
	)
	true_sum_1 = K.sum(
		K.cast(K.equal(y_true_bin, 1.), K.floatx())
	)
	true_sum = K.concatenate([true_sum_0, true_sum_1])
	
	precision = tp_sum / pred_sum
	precision *= K.cast(tf.is_inf(precision), K.floatx())
	
	recall = tp_sum / true_sum
	recall *= K.cast(tf.is_inf(recall), K.floatx())
	
	f1 = (2 * precision * recall) / (precision + recall)
	f1 *= K.cast(tf.is_inf(f1), K.floatx())
	
	# Weighted Average
	return K.sum(f1 * true_sum) / K.sum(true_sum)
	