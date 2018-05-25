'''
restored a model and run session
@author: Kemeng Chen
'''
import tensorflow as tf 
import numpy as np 

class restored_model(object):

	def __init__(self, model_name, model_folder):
		self.graph=tf.Graph()
		self.sess=tf.Session(graph=self.graph)
		print('Read model: ', model_name)

		with self.graph.as_default():
			self.model_saver=tf.train.import_meta_graph(model_name)
			self.model_saver.restore(self.sess, tf.train.latest_checkpoint(model_folder+'/.'))
			self.graph=self.graph
			self.sample_in=self.graph.get_tensor_by_name('sample:0')
			self.c_mask_out=self.graph.get_tensor_by_name('c_mask:0')

	def run_sess(self, patches):
		feed_dict={self.sample_in: patches}
		generated_mask=self.sess.run([self.c_mask_out], feed_dict)
		return generated_mask

	def close_sess(self):
		self.sess.close()