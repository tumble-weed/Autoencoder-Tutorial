import numpy as np
import scipy
from scipy import io as sp_io

import skimage
import sklearn
from sklearn import preprocessing,cross_validation

import os,pdb

import theano
from theano import tensor as T

import custom_io
from custom_io.utils import tile_raster_images as show_row_vectors
from custom_io.imsave2 import imsave2
from custom_io.imsave2 import savefig2
from custom_io.utils import plot_together

from collections import OrderedDict

Ols_mat_name='IMAGES_RAW.mat'
Ols_dict=sp_io.loadmat(Ols_mat_name)
Ols_images=Ols_dict['IMAGESr']

patch_size=(16,16)
Ols_patches=[skimage.util.view_as_windows(im,patch_size,step=4) for im in np.rollaxis(Ols_images,-1,0)]
Ols_patches=np.array(Ols_patches)
n_patches=np.prod(Ols_patches.shape[:3])
nvis=np.prod(Ols_patches.shape[3:])
Ols_patches=Ols_patches.reshape((n_patches,nvis))


random_idx=np.random.permutation(n_patches)
nsamples=40000
kept_idx=random_idx[:nsamples]
kept_patches=Ols_patches[kept_idx,:]

### preprocessing the data
Scaler=preprocessing.StandardScaler(copy=False,with_std=0.33)
processed_patches=Scaler.fit_transform(kept_patches)
processed_patches[processed_patches>1]=1
processed_patches[processed_patches<-1]=-1
processed_patches=(processed_patches+1)*0.4+0.1

## division of train,valid,test sets
train_,valid_test_=cross_validation.train_test_split(processed_patches,
	test_size=0.3,random_state=0)
valid_,test_=cross_validation.train_test_split(valid_test_,test_size=0.5,random_state=0)


##
train=theano.shared(train_)
test=theano.shared(test_)
valid=theano.shared(valid_)

###

nhid=100
W_shape=nhid,nvis
lim=np.sqrt(6./(2*nvis+1))
W_init=np.random.uniform(-lim,lim,W_shape)
W=theano.shared(W_init)

hbias=theano.shared(np.zeros((nhid,1)),broadcastable=[False,True])

U_shape=nvis,nhid
lim1=np.sqrt(6./(2*nhid+1))
U_init=np.random.uniform(-lim1,lim1,U_shape)
U=theano.shared(U_init)

vbias=theano.shared(np.zeros((nvis,1)),broadcastable=[False,True])

###
x_row=T.matrix('x_row')
x=x_row.T

activity_up=T.dot(W,x)+hbias
hid_layer=T.nnet.sigmoid(activity_up)

activity_down=T.dot(U,hid_layer)+vbias
recons=T.nnet.sigmoid(activity_down)

###
cross_entropy=lambda a,b:-a*T.log(b)-(1-a)*T.log(1-b)
cost_recons= T.sum((x_row-recons.T)**2,axis=1)

mean_activity=T.mean(hid_layer,axis=1)
rho=0.05
cost_sparsity=cross_entropy(rho,mean_activity)
cost=T.mean(cost_recons)+T.mean(cost_sparsity)

###
train_updates=OrderedDict({})
params=[W,U,hbias,vbias]
lr=T.scalar('lr',dtype='float64')
for p in params:
	grad_wrt_param=T.grad(cost,p)
	train_updates[p]=p-lr*grad_wrt_param

###
validation_fn=theano.function([],cost,givens={x_row:valid})
reconstruct_validation=theano.function([],recons,givens={x_row:valid})

index=T.scalar('index',dtype='int64')
minibatch_size=128
train_fn=theano.function([index,lr],cost,givens={x_row:train[index*minibatch_size:(index+1)*minibatch_size],U:W.T},updates=train_updates)

###
n_epochs=200
n_batches=train_.shape[0]/minibatch_size

if train_.shape[0]%minibatch_size!=0:
	minibatch_size+=1
save_dir='AE_tutorial'
lr_=0.01
validation_freq=5
n_viz_samples=100
x_tile_shape= (int(np.sqrt(n_viz_samples)),int(np.sqrt(n_viz_samples))+1)
W_tile_shape=(int(np.sqrt(nhid)),int(np.sqrt(nhid))+1)

cost_train_so_far=[]
cost_validation_so_far=[]

for epoch in range(n_epochs):
	epoch_cost=0
	print 'Epoch %d'%epoch
	for batch in range(n_batches):
		batch_cost=train_fn(batch,lr_)
		epoch_cost+=batch_cost
	epoch_cost/=n_batches
	if epoch==0:
		original=valid_[:nsamples]
		img_original=show_row_vectors(original,tile_shape=x_tile_shape,tile_spacing=(2,2),img_shape=patch_size)
		recons_dir=os.path.join(save_dir,'recons')
		imsave2(os.path.join(recons_dir,'original.png'),img_original)

	if (epoch+1)%validation_freq==0:
		validation_cost=validation_fn()
		recons=reconstruct_validation()
		img_recons=show_row_vectors(recons.T,tile_shape=x_tile_shape,tile_spacing=(2,2),img_shape=patch_size)
		imsave2(os.path.join(recons_dir,'%d.png'%epoch),img_recons)

		W_=W.get_value()
		img_W_=show_row_vectors(W_,tile_shape=W_tile_shape,tile_spacing=(2,2),img_shape=patch_size)
		imsave2(os.path.join(save_dir,'filters','%d.png'%epoch),img_W_)

		cost_train_so_far+=[epoch_cost]
		cost_validation_so_far+=[validation_cost]
		x_points=range(len(cost_train_so_far))
		F=plot_together([(x_points,cost_train_so_far),(x_points,cost_validation_so_far)],legends=['train','validation'])
		savefig2(os.path.join(save_dir,'error_plots','%d.png'%epoch))