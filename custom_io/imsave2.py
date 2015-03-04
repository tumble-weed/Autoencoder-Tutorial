import os
from skimage import io
import pylab as P
import pdb
def imsave2(save_path,im):
	dir_tree=list(os.path.split(save_path))
	save_name=dir_tree[1]
	dir_tree=dir_tree[0]
	try:
		os.makedirs(os.path.join(dir_tree))
	except Exception as e:
		#pdb.set_trace()
		#print e
		pass
	io.imsave(save_path,im)

def savefig2(save_path):
	dir_tree=list(os.path.split(save_path))
	save_name=dir_tree[1]
	dir_tree=dir_tree[0]
	try:
		os.makedirs(os.path.join(dir_tree))
	except Exception as e:
		#pdb.set_trace()
		#print e
		pass
	P.savefig(save_path)

