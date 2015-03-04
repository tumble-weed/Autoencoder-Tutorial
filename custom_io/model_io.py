import cPickle
import pdb,os
import numpy as np
from os import makedirs
import theano
from theano import tensor as T

def write_model(model_path,shared_params,other_params=None):

	d,filename=os.path.split(model_path)
	try:
		os.makedirs(d)
	except Exception as e:
		print e
		pass

	f=open(model_path,'wb')
	for p in shared_params:
		cPickle.dump(p.get_value(),f)
	if not other_params == None:
		for p in other_params:
			cPickle.dump(p,f)
	f.close()

def read_model(model_path,shared_params):
	f=open(model_path,'rb')
	for p in shared_params:
		#pdb.set_trace()
		v1=cPickle.load(f)
		p.set_value(v1)
	other_params=[]	

	try:
		while 1:
			other_params.append(cPickle.load(f))
	except Exception as e:
		print e
		pass
	f.close()
	return other_params

# n_filters=60
# chan=1
# H_r,H_c=15,15
# H_shape=(n_filters,chan,H_r,H_c)

# H=theano.shared(value=np.zeros(H_shape))
# b_h=theano.shared(value=np.zeros((1,n_filters,1,1),dtype='float64'),broadcastable=[True,False,True,True])
# b_v=theano.shared(value=np.zeros((1,chan,1,1)),broadcastable=[True,False,True,True])
# shared_params=[H,b_h,b_v]
# read_model('conv_autoencoder_finger2/auto_enc_epoch_5.model',shared_params)
# pdb.set_trace()