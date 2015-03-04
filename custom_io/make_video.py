import os
import sys

im_dir='/home/aniket/vision/code/sift/learn_sift/learn_edgemaps_see_orientations_64/base_orientations'
tmp_dir='tmp/'
full_tmp_dir=os.path.join(im_dir,tmp_dir)
file_list=os.listdir(im_dir)
im_list=[f for f in file_list if f.split('.')[-1] in ['jpg','png','jpeg'] and f.split('.')[0].isdigit() ]
try:
	os.mkdir(full_tmp_dir)
except:
	pass
num_im=len(im_list)
for im_name in im_list:
	new_im_name='%04d.png'%int(im_name.split('.')[0])
	full_im_name=os.path.join(im_dir,im_name)
	full_new_im_name=os.path.join(full_tmp_dir,new_im_name)
	os.system('cp %s %s'%(full_im_name,full_new_im_name))
this_dir=os.getcwd()
os.chdir(full_tmp_dir)
os.system('convert *.png -delay 10 -morph 10 \%05d.morph.jpg')
os.system('ffmpeg -r 25 -qscale 2 -i \%05d.morph.jpg ../output.mp4')
os.chdir(this_dir)
os.system('rm -rf %s'%full_tmp_dir)
