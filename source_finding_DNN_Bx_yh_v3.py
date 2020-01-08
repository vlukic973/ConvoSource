
"""
This script trains and tests AutoSource on the segmented real maps and solutions at a chosen exposure time, frequency and SNR on the simulated SKA data. Run 'generate_real_data_and_solutions_Bx_yh_v1.py' first before running this script. The script commands as they are currently assume there are 50x50 pixel segmented maps space 50 pixels apart. The following commands run AutoSource on the 560MHz data at 8h exposure time, at an SNR of 1.

Usage:
       ... ...
python source_finding_DNN_Bx_yh_v3.py --img_inc 50 --img_size 50 --freq 1 --bg_fits '/path/to/background_fits/SKAMid_B1_8h_pbf_corrected_v3.pybdsm.rmsd_I.fits' --save_dir '/path/where/to/save/results/' --augment 'None' --epochs 50 --load_images '/path/to/segmented/images/real_images_50_50_1_final.npy' --load_solutions '/path/to/segmented/solutions/solutions_50_50_1_final.npy' --snr 1 

python source_finding_DNN_Bx_yh_v3.py --img_inc 50 --img_size 50 --freq 1 --bg_fits '/path/to/background_fits/SKAMid_B1_8h_pbf_corrected_v3.pybdsm.rmsd_I.fits' --save_dir '/path/where/to/save/results/' --augment 'Extended' --epochs 50 --load_images '/path/to/segmented/images/real_images_50_50_1_final.npy' --load_solutions '/path/to/segmented/solutions/solutions_50_50_1_final.npy' --snr 1

python source_finding_DNN_Bx_yh_v3.py --img_inc 50 --img_size 50 --freq 1 --bg_fits '/path/to/background_fits/SKAMid_B1_8h_pbf_corrected_v3.pybdsm.rmsd_I.fits' --save_dir '/path/where/to/save/results/' --augment 'All' --epochs 50 --load_images '/path/to/segmented/images/real_images_50_50_1_final.npy' --load_solutions '/path/to/segmented/solutions/solutions_50_50_1_final.npy' --snr 1

           
Author: Vesna Lukic, E-mail: `vlukic973@gmail.com`
"""


from __future__ import division
import numpy as np
import scipy
import scipy.misc
import scipy.stats
from astropy.stats import sigma_clip
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
from astropy.io import fits
from keras.utils.vis_utils import plot_model
import pandas as pd
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics.pairwise import euclidean_distances
import operator
import skimage
from skimage.transform import warp
from astropy.io import fits
import scipy
from scipy import ndimage
import datetime

from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
from keras.models import Sequential, Model
from keras.utils.vis_utils import plot_model
import numpy as np

import os
import keras
import numpy as np
import keras.backend as K

from scipy.misc import imread
from sklearn.metrics import accuracy_score, normalized_mutual_info_score

import numpy as np
import time
from sklearn.cluster import KMeans
from keras import callbacks
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Dense, Input
from keras.initializers import VarianceScaling
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Dropout, Flatten
from keras import regularizers
from keras import optimizers

from scipy.misc import imread
from sklearn.metrics import accuracy_score, normalized_mutual_info_score

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
import glob

def train(train_X,train_Y,test_X,test_Y):
    """
Training
    """

    input_img = Input(shape=(50, 50, 1))  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (7, 7), strides=1, activation='relu', padding='same')(input_img)
    x = Dropout(0.25)(x)
    x = Conv2D(32, (5, 5), strides=1, activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(x)
    decoded = Dense(1,activation='sigmoid')(x)

    autoencoder = Model(input_img, decoded)

    autoencoder.summary()

    es=keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    adadelta = keras.optimizers.adadelta(lr=1.0, decay=0.0, rho=0.99)

    autoencoder.compile(optimizer=adadelta, loss='binary_crossentropy')

    weight_save_callback = ModelCheckpoint(args.save_dir+'weights_B'+str(args.freq)+'_'+str(args.img_inc)+'_'+str(args.img_size)+'_.{epoch:02d}_loss-{val_loss:.4f}'+str(args.augment)+'.hdf5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

    print(weight_save_callback)

    start = time.time()
    autoencoder_locs=autoencoder.fit(train_X, train_Y, epochs=args.epochs, batch_size=128, shuffle=True, validation_data=(test_X, test_Y),callbacks=[es,weight_save_callback])
    end = time.time()

    print(end-start)

    loss = autoencoder_locs.history['loss']
    val_loss = autoencoder_locs.history['val_loss']
    epochs = range(len(loss))

    np.save(args.save_dir+datetime.datetime.now().isoformat()[0:18]+'_loss.npy',loss)
    np.save(args.save_dir+datetime.datetime.now().isoformat()[0:18]+'_val_loss.npy',val_loss)
    np.save(args.save_dir+datetime.datetime.now().isoformat()[0:18]+'epochs.npy',epochs)

    return loss,val_loss,epochs

def test(test_X,test_Y,class_test):
    """
Testing
    """

    input_img = Input(shape=(50, 50, 1))  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (7, 7), strides=1, activation='relu', padding='same')(input_img)
    x = Dropout(0.25)(x)
    x = Conv2D(32, (5, 5), strides=1, activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(x)
    decoded = Dense(1,activation='sigmoid')(x)

    autoencoder = Model(input_img, decoded)

    autoencoder.summary()

    es=keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    adadelta = keras.optimizers.adadelta(lr=1.0, decay=0.0, rho=0.99)

    autoencoder.compile(optimizer=adadelta, loss='binary_crossentropy')

    if (args.augment=='None'):

	    list_of_files = glob.glob(args.save_dir+'/*None.hdf5') # * means all if need specific format then *.csv
	    latest_file = max(list_of_files, key=os.path.getctime)

    if (args.augment=='Extended'):

	    list_of_files = glob.glob(args.save_dir+'/*Extended.hdf5') # * means all if need specific format then *.csv
	    latest_file = max(list_of_files, key=os.path.getctime)

    if (args.augment=='All'):

	    list_of_files = glob.glob(args.save_dir+'/*All.hdf5') # * means all if need specific format then *.csv
	    latest_file = max(list_of_files, key=os.path.getctime)


    autoencoder.load_weights(latest_file)

    reconstructed = Model(inputs=autoencoder.input, outputs=autoencoder.output)
    first_layer_activation = Model(inputs=autoencoder.input, outputs=autoencoder.layers[1].output)
    second_layer_activation = Model(inputs=autoencoder.input, outputs=autoencoder.layers[2].output)
    third_layer_activation = Model(inputs=autoencoder.input, outputs=autoencoder.layers[3].output)
    fourth_layer_activation = Model(inputs=autoencoder.input, outputs=autoencoder.layers[4].output)
    fifth_layer_activation = Model(inputs=autoencoder.input, outputs=autoencoder.layers[5].output)

    recon_image=reconstructed.predict(test_X)

    first_layer_out=first_layer_activation.predict(test_X)
    second_layer_out=second_layer_activation.predict(test_X)
    third_layer_out=third_layer_activation.predict(test_X)
    fourth_layer_out=fourth_layer_activation.predict(test_X)
    fifth_layer_out=fifth_layer_activation.predict(test_X)

    recon_image=recon_image.reshape(recon_image.shape[0],recon_image.shape[1],recon_image.shape[2],1)

    print(class_test.shape) 
    print(np.max(recon_image))

    if (args.show_img=='True'):
        print('Now showing images and detected features')
	plt.ion()
	for i in np.where(class_test==1)[0][0:10]:
	    	plt.subplot(1,3,1); plt.axis('off'); plt.imshow(test_X[i,:,:,0])
	    	plt.subplot(1,3,2); plt.axis('off'); plt.imshow(class_test[i,:,:,0])
	    	plt.subplot(1,3,3); plt.axis('off'); plt.imshow(recon_image[i,:,:,0])
		plt.savefig(args.save_dir+'test_X_Y_recon'+str(i)+'.png')
		plt.close()

    precision_sfg_list=[]
    recall_sfg_list=[]
    F1_score_sfg_list=[]
    accuracy_sfg_list=[]

    precision_ss_list=[]
    recall_ss_list=[]
    F1_score_ss_list=[]
    accuracy_ss_list=[]

    precision_fs_list=[]
    recall_fs_list=[]
    F1_score_fs_list=[]
    accuracy_fs_list=[]

    precision_all_list=[]
    recall_all_list=[]
    F1_score_all_list=[]
    accuracy_all_list=[]

    for recon_threshold in np.round(np.linspace(0.1,0.99,11),2):

	print(recon_threshold)

	array_loc=[]
	true_loc=[]
	class_ss=[]
	class_fs=[]
	class_sfg=[]
	class_all=[]
	sum_source=[]

	for i in range(0,len(recon_image)):
		array_loc.append(np.where(recon_threshold*np.max(recon_image[i,:,:,0])<recon_image[i,:,:,0]))
		class_sfg.append(np.where(class_test[i,:,:,0]==3))
		class_ss.append(np.where(class_test[i,:,:,0]==1)) 
		class_fs.append(np.where(class_test[i,:,:,0]==2)) 
		class_all.append(np.where(test_Y[i,:,:,0]==1))
		sum_source.append(np.sum(class_test[i]))

	pred_loc=[]
	class_sfg1=[]
	class_ss1=[]	
	class_fs1=[]
	class_all1=[]	

	for i in range(0,len(recon_image)):
		pred_loc.append(zip(array_loc[i][0],array_loc[i][1]))
		class_sfg1.append(zip(class_sfg[i][0],class_sfg[i][1]))
		class_ss1.append(zip(class_ss[i][0],class_ss[i][1]))
		class_fs1.append(zip(class_fs[i][0],class_fs[i][1]))
		class_all1.append(zip(class_all[i][0],class_all[i][1]))		

	pred_sfg=[i for i,x in enumerate(class_sfg1) if x]
	pred_ss=[i for i,x in enumerate(class_ss1) if x]
	pred_fs=[i for i,x in enumerate(class_fs1) if x]	
	pred_all=[i for i,x in enumerate(class_all1) if x]

	pred_sfg1 = [pred_loc[i] for i in pred_sfg]
	pred_ss1 = [pred_loc[i] for i in pred_ss]
	pred_fs1 = [pred_loc[i] for i in pred_fs]
	pred_all1 = [pred_loc[i] for i in pred_all]

	class_sfg2 = [class_sfg1[i] for i in pred_sfg]
	class_ss2 = [class_ss1[i] for i in pred_ss]
	class_fs2 = [class_fs1[i] for i in pred_fs]
	class_all2 = [class_all1[i] for i in pred_all]

	euc_dist_sfg=[]
	euc_dist_ss=[]
	euc_dist_fs=[]
	euc_dist_all=[]

	for i in range(0,len(pred_sfg1)):
		if (pred_sfg1[i]!=[]):
			euc_dist_sfg.append(np.round(euclidean_distances(pred_sfg1[i],pred_sfg1[i])))
		else:
			euc_dist_sfg.append([])

	for i in range(0,len(pred_ss1)):
		if (pred_ss1[i]!=[]):
			euc_dist_ss.append(np.round(euclidean_distances(pred_ss1[i],pred_ss1[i])))
		else:
			euc_dist_ss.append([])

	for i in range(0,len(pred_fs1)):
		if (pred_fs1[i]!=[]):
			euc_dist_fs.append(np.round(euclidean_distances(pred_fs1[i],pred_fs1[i])))
		else:
			euc_dist_fs.append([])

	for i in range(0,len(pred_all1)):
		if (pred_all1[i]!=[]):
			euc_dist_all.append(np.round(euclidean_distances(pred_all1[i],pred_all1[i])))
		else:
			euc_dist_all.append([])

	t=1

	des_elem=[]
	des_elem2=[]
	des_elem_sfg=[]

	for i in range(0,len(pred_sfg1)):
		for j in range(1,len(euc_dist_sfg[i])):
			des_elem.append((0))
			if (euc_dist_sfg[i].item((j,j-t))>1):
				des_elem.append(j)
		des_elem2.append(des_elem)
		des_elem_sfg.append(list(set(des_elem2[0])))
		des_elem=[]
		des_elem2=[]

	for n, i in enumerate(des_elem_sfg):
		if i == []:
			des_elem_sfg[n] = [0]

	ind_loc=[]
	ind_loc_sfg=[]

	for i in range(0,len(pred_sfg1)):
		for j in range(0,len(des_elem_sfg[i])):
			if (pred_sfg1[i]!=[]):
				ind_loc.append(pred_sfg1[i][des_elem_sfg[i][j]])
			else:
				ind_loc.append([])
		ind_loc_sfg.append(ind_loc)
		ind_loc=[]

	df_sfg=pd.DataFrame()

	df_sfg['pred_loc']=pred_sfg1
	df_sfg['des_elem_sfg']=des_elem_sfg
	df_sfg['ind_loc2']=ind_loc_sfg

	t=1

	des_elem=[]
	des_elem2=[]
	des_elem_ss=[]

	for i in range(0,len(pred_ss1)):
		for j in range(1,len(euc_dist_ss[i])):
			des_elem.append((0))
			if (euc_dist_ss[i].item((j,j-t))>1):
				des_elem.append(j)
		des_elem2.append(des_elem)
		des_elem_ss.append(list(set(des_elem2[0])))
		des_elem=[]
		des_elem2=[]

	for n, i in enumerate(des_elem_ss):
		if i == []:
			des_elem_ss[n] = [0]

	ind_loc=[]
	ind_loc_ss=[]

	for i in range(0,len(pred_ss1)):
		for j in range(0,len(des_elem_ss[i])):
			if (pred_ss1[i]!=[]):
				ind_loc.append(pred_ss1[i][des_elem_ss[i][j]])
			else:
				ind_loc.append([])
		ind_loc_ss.append(ind_loc)
		ind_loc=[]
	
	df_ss=pd.DataFrame()

	df_ss['pred_loc']=pred_ss1
	df_ss['des_elem_ss']=des_elem_ss
	df_ss['ind_loc2']=ind_loc_ss

	t=1

	des_elem=[]
	des_elem2=[]
	des_elem_fs=[]

	for i in range(0,len(pred_fs1)):
		for j in range(1,len(euc_dist_fs[i])):
			des_elem.append((0))
			if (euc_dist_fs[i].item((j,j-t))>1):
				des_elem.append(j)
		des_elem2.append(des_elem)
		des_elem_fs.append(list(set(des_elem2[0])))
		des_elem=[]
		des_elem2=[]

	for n, i in enumerate(des_elem_fs):
		if i == []:
			des_elem_fs[n] = [0]

	ind_loc=[]
	ind_loc_fs=[]

	for i in range(0,len(pred_fs1)):
		for j in range(0,len(des_elem_fs[i])):
			if (pred_fs1[i]!=[]):
				ind_loc.append(pred_fs1[i][des_elem_fs[i][j]])
			else:
				ind_loc.append([])
		ind_loc_fs.append(ind_loc)
		ind_loc=[]

	df_fs=pd.DataFrame()

	df_fs['pred_loc']=pred_fs1
	df_fs['des_elem_fs']=des_elem_fs
	df_fs['ind_loc2']=ind_loc_fs

	t=1

	des_elem=[]
	des_elem2=[]
	des_elem_all=[]

	for i in range(0,len(pred_all1)):
		for j in range(1,len(euc_dist_all[i])):
			des_elem.append((0))
			if (euc_dist_all[i].item((j,j-t))>1):
				des_elem.append(j)
		des_elem2.append(des_elem)
		des_elem_all.append(list(set(des_elem2[0])))
		des_elem=[]
		des_elem2=[]

	for n, i in enumerate(des_elem_all):
		if i == []:
			des_elem_all[n] = [0]

	ind_loc=[]
	ind_loc_all=[]

	for i in range(0,len(pred_all1)):
		for j in range(0,len(des_elem_all[i])):
			if (pred_all1[i]!=[]):
				ind_loc.append(pred_all1[i][des_elem_all[i][j]])
			else:
				ind_loc.append([])
		ind_loc_all.append(ind_loc)
		ind_loc=[]

	df_all=pd.DataFrame()

	df_all['pred_loc']=pred_all1
	df_all['des_elem_all']=des_elem_all
	df_all['ind_loc2']=ind_loc_all

	class_ss=[]
	class_ss_shape=[]
	class_sfg=[]
	class_sfg_shape=[]
	class_fs=[]
	class_fs_shape=[]
	class_all=[]
	class_all_shape=[]

	for i in range(0,len(class_sfg2)):
		if ((class_sfg2[i]!=[]) & (ind_loc_sfg[i]!=[[]])):
			class_sfg.append(np.round(euclidean_distances(ind_loc_sfg[i],class_sfg2[i])))
			class_sfg_shape.append(euclidean_distances(ind_loc_sfg[i],class_sfg2[i]).shape)
		else:
			class_sfg.append([])
			class_sfg_shape.append([])

	for i in range(0,len(class_ss2)):
		if ((class_ss2[i]!=[]) & (ind_loc_ss[i]!=[[]])):
			class_ss.append(np.round(euclidean_distances(ind_loc_ss[i],class_ss2[i])))
			class_ss_shape.append(euclidean_distances(ind_loc_ss[i],class_ss2[i]).shape)
		else:
			class_ss.append([])
			class_ss_shape.append([])

	for i in range(0,len(class_fs2)):
		if ((class_fs2[i]!=[]) & (ind_loc_fs[i]!=[[]])):
			class_fs.append(np.round(euclidean_distances(ind_loc_fs[i],class_fs2[i])))
			class_fs_shape.append(euclidean_distances(ind_loc_fs[i],class_fs2[i]).shape)
		else:
			class_fs.append([])
			class_fs_shape.append([])

	for i in range(0,len(class_all2)):
		if ((class_all2[i]!=[]) & (ind_loc_all[i]!=[[]])):
			class_all.append(np.round(euclidean_distances(ind_loc_all[i],class_all2[i])))
			class_all_shape.append(euclidean_distances(ind_loc_all[i],class_all2[i]).shape)
		else:
			class_all.append([])
			class_all_shape.append([])

	TP_sfg_list=[]
	TP_ss_list=[]
	TP_fs_list=[]
	TP_all_list=[]
	FN_sfg_list=[]
	FN_ss_list=[]
	FN_fs_list=[]
	FN_all_list=[]
	FP_sfg_list2=[]
	FP_ss_list2=[]
	FP_fs_list2=[]
	FP_all_list2=[]
	pred_sfg_loc_len=[]
	pred_ss_loc_len=[]
	pred_fs_loc_len=[]
	pred_all_loc_len=[]
	class_loc_len=[]

	pix_threshold=3

	for i in range(0,len(class_sfg2)):
		TP_sfg_list.append(np.sum(class_sfg[i]<pix_threshold))
		pred_sfg_loc_len.append(len(pred_sfg1[i]))
		
	for i in range(0,len(class_ss2)):
		TP_ss_list.append(np.sum(class_ss[i]<pix_threshold))		
		pred_ss_loc_len.append(len(pred_ss1[i]))

	for i in range(0,len(class_fs2)):
		TP_fs_list.append(np.sum(class_fs[i]<pix_threshold))		
		pred_fs_loc_len.append(len(pred_fs1[i]))

	for i in range(0,len(class_all2)):
		TP_all_list.append(np.sum(class_all[i]<pix_threshold))		
		pred_all_loc_len.append(len(pred_all1[i]))

	for i in range(0,len(class_sfg2)):
		if (class_sfg_shape[i]!=[]):
			FN_sfg_list.append(class_sfg_shape[i][1]-class_sfg_shape[i][0])
			FP_sfg_list2.append(class_sfg_shape[i][0]-class_sfg_shape[i][1])
		else:
			FN_sfg_list.append(0)
			FP_sfg_list2.append(0)

	for i in range(0,len(class_ss2)):
		if (class_ss_shape[i]!=[]):
			FN_ss_list.append(class_ss_shape[i][1]-class_ss_shape[i][0])
			FP_ss_list2.append(class_ss_shape[i][0]-class_ss_shape[i][1])
		else:
			FN_ss_list.append(0)
			FP_ss_list2.append(0)

	for i in range(0,len(class_fs2)):
		if (class_fs_shape[i]!=[]):
			FN_fs_list.append(class_fs_shape[i][1]-class_fs_shape[i][0])
			FP_fs_list2.append(class_fs_shape[i][0]-class_fs_shape[i][1])
		else:
			FN_fs_list.append(0)
			FP_fs_list2.append(0)

	for i in range(0,len(class_all2)):
		if (class_all_shape[i]!=[]):
			FN_all_list.append(class_all_shape[i][1]-class_all_shape[i][0])
			FP_all_list2.append(class_all_shape[i][0]-class_all_shape[i][1])
		else:
			FN_all_list.append(0)
			FP_all_list2.append(0)

	i_df=[]
	j_df=[]

	a=-1
	b=-1

	for i in range(0,50,1):
		a+=1
		for j in range(0,50,1):
			b+=1
			if (b==50):
				b=0
				i_df.append(i);j_df.append(j)
			else:
				i_df.append(i);j_df.append(j)

	all_coords=zip(i_df,j_df)
	true_sfg_negatives_list=[]
	true_ss_negatives_list=[]
	true_fs_negatives_list=[]
	true_all_negatives_list=[]

	for i in range(0,len(class_sfg2)):
		not_in_sfg_loc1 = list(set(all_coords) - set(class_sfg1[i]))
		not_in_sfg_pred_loc = list(set(all_coords) - set(pred_sfg1[i]))
		true_sfg_negatives=len(list(set(not_in_sfg_loc1).intersection(not_in_sfg_pred_loc)))
		true_sfg_negatives_list.append(true_sfg_negatives)

	for i in range(0,len(class_ss2)):
		not_in_ss_loc1 = list(set(all_coords) - set(class_ss1[i]))
		not_in_ss_pred_loc = list(set(all_coords) - set(pred_ss1[i]))
		true_ss_negatives=len(list(set(not_in_ss_loc1).intersection(not_in_ss_pred_loc)))
		true_ss_negatives_list.append(true_ss_negatives)

	for i in range(0,len(class_fs2)):
		not_in_fs_loc1 = list(set(all_coords) - set(class_fs1[i]))
		not_in_fs_pred_loc = list(set(all_coords) - set(pred_fs1[i]))
		true_fs_negatives=len(list(set(not_in_fs_loc1).intersection(not_in_fs_pred_loc)))
		true_fs_negatives_list.append(true_fs_negatives)

	for i in range(0,len(class_all2)):
		not_in_all_loc1 = list(set(all_coords) - set(class_all1[i]))
		not_in_all_pred_loc = list(set(all_coords) - set(pred_all1[i]))
		true_all_negatives=len(list(set(not_in_all_loc1).intersection(not_in_all_pred_loc)))
		true_all_negatives_list.append(true_all_negatives)

	df_sfg=pd.DataFrame()

	df_sfg['euc_dist_shape']=class_sfg_shape
	df_sfg['pred_sfg_loc_len']=pred_sfg_loc_len
	df_sfg['TP_sfg']=TP_sfg_list
	df_sfg['FP_sfg_list1']=df_sfg['pred_sfg_loc_len']-df_sfg['TP_sfg']
	df_sfg.loc[df_sfg['FP_sfg_list1'] < 0, 'FP_sfg_list1'] = 0
	df_sfg['FN_sfg']=FN_sfg_list
	df_sfg['FP_sfg_list2']=FP_sfg_list2
	df_sfg.loc[df_sfg['FP_sfg_list2'] < 0, 'FP_sfg_list2'] = 0
	df_sfg.loc[df_sfg['FN_sfg'] < 0, 'FN_sfg'] = 0
	df_sfg['FP_sfg']=df_sfg['FP_sfg_list1']+df_sfg['FP_sfg_list2']
	df_sfg['TN_sfg']=true_sfg_negatives_list

	df_ss=pd.DataFrame()

	df_ss['euc_dist_shape']=class_ss_shape
	df_ss['pred_ss_loc_len']=pred_ss_loc_len
	df_ss['TP_ss']=TP_ss_list
	df_ss['FP_ss_list1']=df_ss['pred_ss_loc_len']-df_ss['TP_ss']
	df_ss.loc[df_ss['FP_ss_list1'] < 0, 'FP_ss_list1'] = 0
	df_ss['FN_ss']=FN_ss_list
	df_ss['FP_ss_list2']=FP_ss_list2
	df_ss.loc[df_ss['FP_ss_list2'] < 0, 'FP_ss_list2'] = 0
	df_ss.loc[df_ss['FN_ss'] < 0, 'FN_ss'] = 0
	df_ss['FP_ss']=df_ss['FP_ss_list1']+df_ss['FP_ss_list2']
	df_ss['TN_ss']=true_ss_negatives_list

	df_fs=pd.DataFrame()

	df_fs['euc_dist_shape']=class_fs_shape
	df_fs['pred_fs_loc_len']=pred_fs_loc_len
	df_fs['TP_fs']=TP_fs_list
	df_fs['FP_fs_list1']=df_fs['pred_fs_loc_len']-df_fs['TP_fs']
	df_fs.loc[df_fs['FP_fs_list1'] < 0, 'FP_fs_list1'] = 0
	df_fs['FN_fs']=FN_fs_list
	df_fs['FP_fs_list2']=FP_fs_list2
	df_fs.loc[df_fs['FP_fs_list2'] < 0, 'FP_fs_list2'] = 0
	df_fs.loc[df_fs['FN_fs'] < 0, 'FN_fs'] = 0
	df_fs['FP_fs']=df_fs['FP_fs_list1']+df_fs['FP_fs_list2']
	df_fs['TN_fs']=true_fs_negatives_list

	df_all=pd.DataFrame()

	df_all['euc_dist_shape']=class_all_shape
	df_all['pred_all_loc_len']=pred_all_loc_len
	df_all['TP_all']=TP_all_list
	df_all['FP_all_list1']=df_all['pred_all_loc_len']-df_all['TP_all']
	df_all.loc[df_all['FP_all_list1'] < 0, 'FP_all_list1'] = 0
	df_all['FN_all']=FN_all_list
	df_all['FP_all_list2']=FP_all_list2
	df_all.loc[df_all['FP_all_list2'] < 0, 'FP_all_list2'] = 0
	df_all.loc[df_all['FN_all'] < 0, 'FN_all'] = 0
	df_all['FP_all']=df_all['FP_all_list1']+df_all['FP_all_list2']
	df_all['TN_all']=true_all_negatives_list


	precision_sfg=np.sum(df_sfg['TP_sfg'])/(np.sum(df_sfg['TP_sfg'])+np.sum(df_sfg['FP_sfg'])+0.00001)
	recall_sfg=np.sum(df_sfg['TP_sfg'])/(np.sum(df_sfg['TP_sfg'])+np.sum(df_sfg['FN_sfg'])+0.00001)
	F1_score_sfg=2*precision_sfg*recall_sfg/(precision_sfg+recall_sfg+0.00001)
	accuracy_sfg=(np.sum(df_sfg['TP_sfg'])+np.sum(df_sfg['TN_sfg']))/(np.sum(df_sfg['TP_sfg'])+np.sum(df_sfg['TN_sfg'])+np.sum(df_sfg['FP_sfg'])+np.sum(df_sfg['FN_sfg'])+0.00001)

	precision_ss=np.sum(df_ss['TP_ss'])/(np.sum(df_ss['TP_ss'])+np.sum(df_ss['FP_ss'])+0.00001)
	recall_ss=np.sum(df_ss['TP_ss'])/(np.sum(df_ss['TP_ss'])+np.sum(df_ss['FN_ss'])+0.00001)
	F1_score_ss=2*precision_ss*recall_ss/(precision_ss+recall_ss+0.00001)
	accuracy_ss=(np.sum(df_ss['TP_ss'])+np.sum(df_ss['TN_ss']))/(np.sum(df_ss['TP_ss'])+np.sum(df_ss['TN_ss'])+np.sum(df_ss['FP_ss'])+np.sum(df_ss['FN_ss'])+0.00001)

	precision_fs=np.sum(df_fs['TP_fs'])/(np.sum(df_fs['TP_fs'])+np.sum(df_fs['FP_fs'])+0.00001)
	recall_fs=np.sum(df_fs['TP_fs'])/(np.sum(df_fs['TP_fs'])+np.sum(df_fs['FN_fs'])+0.00001)
	F1_score_fs=2*precision_fs*recall_fs/(precision_fs+recall_fs+0.00001)
	accuracy_fs=(np.sum(df_fs['TP_fs'])+np.sum(df_fs['TN_fs']))/(np.sum(df_fs['TP_fs'])+np.sum(df_fs['TN_fs'])+np.sum(df_fs['FP_fs'])+np.sum(df_fs['FN_fs'])+0.00001)

	precision_all=np.sum(df_all['TP_all'])/(np.sum(df_all['TP_all'])+np.sum(df_all['FP_all'])+0.00001)
	recall_all=np.sum(df_all['TP_all'])/(np.sum(df_all['TP_all'])+np.sum(df_all['FN_all'])+0.00001)
	F1_score_all=2*precision_all*recall_all/(precision_all+recall_all+0.00001)
	accuracy_all=(np.sum(df_all['TP_all'])+np.sum(df_all['TN_all']))/(np.sum(df_all['TP_all'])+np.sum(df_all['TN_all'])+np.sum(df_all['FP_all'])+np.sum(df_all['FN_all'])+0.00001)

	precision_sfg_list.append(precision_sfg)
	recall_sfg_list.append(recall_sfg)
	F1_score_sfg_list.append(F1_score_sfg)
	accuracy_sfg_list.append(accuracy_sfg)

	precision_ss_list.append(precision_ss)
	recall_ss_list.append(recall_ss)
	F1_score_ss_list.append(F1_score_ss)
	accuracy_ss_list.append(accuracy_ss)

	precision_fs_list.append(precision_fs)
	recall_fs_list.append(recall_fs)
	F1_score_fs_list.append(F1_score_fs)
	accuracy_fs_list.append(accuracy_fs)

	precision_all_list.append(precision_all)
	recall_all_list.append(recall_all)
	F1_score_all_list.append(F1_score_all)
	accuracy_all_list.append(accuracy_all)

        index, value = max(enumerate(F1_score_all_list), key=operator.itemgetter(1))

        precision_sfg=precision_sfg_list[index]
        recall_sfg=recall_sfg_list[index]
        F1_score_sfg=F1_score_sfg_list[index]
        accuracy_sfg=accuracy_sfg_list[index]

        precision_ss=precision_ss_list[index]
        recall_ss=recall_ss_list[index]
        F1_score_ss=F1_score_ss_list[index]
        accuracy_ss=accuracy_ss_list[index]

        precision_fs=precision_fs_list[index]
        recall_fs=recall_fs_list[index]
        F1_score_fs=F1_score_fs_list[index]
        accuracy_fs=accuracy_fs_list[index]

        precision_all=precision_all_list[index]
        recall_all=recall_all_list[index]
        F1_score_all=F1_score_all_list[index]
        accuracy_all=accuracy_all_list[index]

    return precision_sfg,recall_sfg,F1_score_sfg,accuracy_sfg,precision_ss,recall_ss,F1_score_ss,accuracy_ss,precision_fs,recall_fs,F1_score_fs,accuracy_fs, precision_all,recall_all,F1_score_all,accuracy_all,np.sum(df_sfg['TP_sfg']),np.sum(df_sfg['FP_sfg']),np.sum(df_sfg['TN_sfg']),np.sum(df_sfg['FN_sfg']),np.sum(df_ss['TP_ss']),np.sum(df_ss['FP_ss']),np.sum(df_ss['TN_ss']),np.sum(df_ss['FN_ss']),np.sum(df_fs['TP_fs']),np.sum(df_fs['FP_fs']),np.sum(df_fs['TN_fs']),np.sum(df_fs['FN_fs']),np.sum(df_all['TP_all']),np.sum(df_all['FP_all']),np.sum(df_all['TN_all']),np.sum(df_all['FN_all']),df_all

def augment_none(args):

    images_orig=np.load(args.load_images)

    class_image=np.load(args.load_solutions)

    data2=class_image

    data2=np.where(data2==2, 1, data2) 
    data2=np.where(data2==3, 1, data2) 

    solutions_orig=data2

    hdul = fits.open(args.bg_fits)

    if (args.freq==1):

	    data=hdul[0].data[0,0,16300:20300,16300:20300]
	    print('frequency is 560MHz')

    if (args.freq==2):
    
	    data=hdul[0].data[0,0,16300:20500,16300:20500]
	    print('frequency is 1400MHz')

    if (args.freq==3):

	    data=hdul[0].data[0,0,21700:25700,16300:20300]
	    print('frequency is 9200MHz')

    data[np.isnan(data)] = 0

    images_orig[np.isnan(images_orig)] = np.mean(data)

    images_orig=images_orig.reshape(images_orig.shape[0],args.img_size,args.img_size)

    images_orig=images_orig*args.mult

    print('Maximum array value is:')
    print(np.max(images_orig))

    images_all=images_orig
    solutions_all=solutions_orig

    images_all=images_all.reshape(images_all.shape[0],images_all.shape[1],images_all.shape[2],1)
    solutions_all=solutions_all.reshape(solutions_all.shape[0],solutions_all.shape[1],solutions_all.shape[2],1)
    class_image=class_image.reshape(class_image.shape[0],class_image.shape[1],class_image.shape[2],1)

    train_proportion=args.train_prop

    train_X=images_all[0:int(train_proportion*len(images_all))]
    test_X=images_all[int(train_proportion*len(images_all)):len(images_all)]
    train_Y=solutions_all[0:int(train_proportion*len(images_all))]
    test_Y=solutions_all[int(train_proportion*len(images_all)):len(images_all)]
    class_train=class_image[0:int(train_proportion*len(images_all))]
    class_test=class_image[int(train_proportion*len(images_all)):len(images_all)]

    print(train_X.shape)
    print(test_X.shape)
    print(train_Y.shape)
    print(test_Y.shape)
    print(class_train.shape)
    print(class_test.shape)

    train_X = train_X.astype('float32')
    train_Y = train_Y.astype('float32')
    test_X = test_X.astype('float32')
    test_Y = test_Y.astype('float32')
    class_train = class_train.astype('float32')
    class_test = class_test.astype('float32')

    return train_X,train_Y,test_X,test_Y,class_train,class_test

def augment(args):

    images_orig=np.load(args.load_images)

    class_image=np.load(args.load_solutions)

    data2=class_image

    data2=np.where(data2==2, 1, data2) 
    data2=np.where(data2==3, 1, data2) 

    solutions_orig=data2

    hdul = fits.open(args.bg_fits)

    if (args.freq==1):

	    data=hdul[0].data[0,0,16300:20300,16300:20300]
	    print('frequency is 560MHz')

    if (args.freq==2):
    
	    data=hdul[0].data[0,0,16300:20500,16300:20500]
	    print('frequency is 1400MHz')

    if (args.freq==3):

	    data=hdul[0].data[0,0,19700:25700,19700:25700]
	    print('frequency is 9200MHz')

    data[np.isnan(data)] = 0

    images_orig[np.isnan(images_orig)] = np.mean(data)

    images_orig=images_orig.reshape(images_orig.shape[0],args.img_size,args.img_size)

    images_orig=images_orig*args.mult

    print('Maximum array value is:')
    print(np.max(images_orig))

    images_all=images_orig
    solutions_all=solutions_orig

    images_all=images_all.reshape(images_all.shape[0],images_all.shape[1],images_all.shape[2],1)
    solutions_all=solutions_all.reshape(solutions_all.shape[0],solutions_all.shape[1],solutions_all.shape[2],1)
    class_image=class_image.reshape(class_image.shape[0],class_image.shape[1],class_image.shape[2],1)

    train_proportion=args.train_prop

    train_X=images_all[0:int(train_proportion*len(images_all))]
    test_X=images_all[int(train_proportion*len(images_all)):len(images_all)]
    train_Y=solutions_all[0:int(train_proportion*len(images_all))]
    test_Y=solutions_all[int(train_proportion*len(images_all)):len(images_all)]
    class_train=class_image[0:int(train_proportion*len(images_all))]
    class_test=class_image[int(train_proportion*len(images_all)):len(images_all)]

    print(train_X.shape)
    print(test_X.shape)
    print(train_Y.shape)
    print(test_Y.shape)
    print(class_train.shape)
    print(class_test.shape)

    train_X = train_X.astype('float32')
    train_Y = train_Y.astype('float32')
    test_X = test_X.astype('float32')
    test_Y = test_Y.astype('float32')
    class_train = class_train.astype('float32')
    class_test = class_test.astype('float32')

    extended_index_train=np.unique(np.concatenate((np.where(class_train==1)[0], np.where(class_train==2)[0]), axis=0))

    train_X_extended=train_X[extended_index_train]
    train_Y_extended=train_Y[extended_index_train]

    if (args.augment=='Extended'):

	    x_augment=train_X_extended
	    y_augment=train_Y_extended

    if (args.augment=='All'):

	    x_augment=train_X
	    y_augment=train_Y

    real_img_lr = np.empty((0,50,50))
    real_img_ud = np.empty((0,50,50))
    real_img_90 = np.empty((0,50,50))
    real_img_180 = np.empty((0,50,50))
    real_img_270 = np.empty((0,50,50))
    solutions_lr = np.empty((0,50,50))
    solutions_ud = np.empty((0,50,50))
    solutions_90 = np.empty((0,50,50))
    solutions_180 = np.empty((0,50,50))
    solutions_270 = np.empty((0,50,50))

    for i in range(0,len(x_augment)):
	print(i)
	real_img_lr = np.append(real_img_lr, np.fliplr(x_augment[i]).reshape(1,50,50),axis=0)
	real_img_ud = np.append(real_img_ud, np.flipud(x_augment[i]).reshape(1,50,50),axis=0)
	real_img_90 = np.append(real_img_90, scipy.ndimage.rotate(x_augment[i],angle=90,reshape=True,cval=0).reshape(1,50,50),axis=0)
	real_img_180 = np.append(real_img_180, scipy.ndimage.rotate(x_augment[i],angle=180,reshape=True,cval=0).reshape(1,50,50),axis=0)
	real_img_270 = np.append(real_img_270, scipy.ndimage.rotate(x_augment[i],angle=270,reshape=True,cval=0).reshape(1,50,50),axis=0)
	solutions_lr = np.append(solutions_lr, np.fliplr(y_augment[i]).reshape(1,50,50),axis=0)
	solutions_ud = np.append(solutions_ud, np.flipud(y_augment[i]).reshape(1,50,50),axis=0)
	solutions_90 = np.append(solutions_90, scipy.ndimage.rotate(y_augment[i],angle=90,reshape=True,cval=0).reshape(1,50,50),axis=0)
	solutions_180 = np.append(solutions_180, scipy.ndimage.rotate(y_augment[i],angle=180,reshape=True,cval=0).reshape(1,50,50),axis=0)
	solutions_270 = np.append(solutions_270, scipy.ndimage.rotate(y_augment[i],angle=270,reshape=True,cval=0).reshape(1,50,50),axis=0)

#train_X=train_X.reshape(train_X.shape[0],train_X.shape[1],train_X.shape[2],1)
#train_Y=train_Y.reshape(train_Y.shape[0],train_Y.shape[1],train_Y.shape[2],1)

    x_augment=x_augment.reshape(x_augment.shape[0],x_augment.shape[1],x_augment.shape[2],1)
    y_augment=y_augment.reshape(y_augment.shape[0],y_augment.shape[1],y_augment.shape[2],1)


    train_X1=np.concatenate((x_augment,real_img_lr.reshape(real_img_lr.shape[0],real_img_lr.shape[1],real_img_lr.shape[2],1),real_img_ud.reshape(real_img_ud.shape[0],real_img_ud.shape[1],real_img_ud.shape[2],1),real_img_90.reshape(real_img_90.shape[0],real_img_90.shape[1],real_img_90.shape[2],1),real_img_180.reshape(real_img_180.shape[0],real_img_180.shape[1],real_img_180.shape[2],1),real_img_270.reshape(real_img_270.shape[0],real_img_270.shape[1],real_img_270.shape[2],1)))
    train_Y1=np.concatenate((y_augment,solutions_lr.reshape(solutions_lr.shape[0],solutions_lr.shape[1],solutions_lr.shape[2],1),solutions_ud.reshape(solutions_ud.shape[0],solutions_ud.shape[1],solutions_ud.shape[2],1),solutions_90.reshape(solutions_90.shape[0],solutions_90.shape[1],solutions_90.shape[2],1),solutions_180.reshape(solutions_180.shape[0],solutions_180.shape[1],solutions_180.shape[2],1),solutions_270.reshape(solutions_270.shape[0],solutions_270.shape[1],solutions_270.shape[2],1)))

    train_X=np.concatenate((train_X,train_X1))
    train_Y=np.concatenate((train_Y,train_Y1))


    return train_X,train_Y,test_X,test_Y,class_train,class_test


if __name__ == "__main__":
    import os
    import argparse
    from keras import callbacks

    parser = argparse.ArgumentParser(description="Run the AutoSource source-finder")
    parser.add_argument('--load_images', default='/path/to/segmented/images/real_images_50_505_final.npy')
    parser.add_argument('--load_solutions', default='/path/to/segmented/solutions/solutions_50_505_final.npy')
    parser.add_argument('--save_dir', default='/path/where/to/save/results/')
    parser.add_argument('--bg_fits', default='/path/to/background_fits/SKAMid_B1_8h_pbf_corrected_v3.pybdsm.rmsd_I.fits')
    parser.add_argument('--freq', default=1,type=int)
    parser.add_argument('--snr', default=5, type=int)
    parser.add_argument('--img_inc', default=50,type=int)
    parser.add_argument('--img_size', default=50,type=int)
    parser.add_argument('--mult', default=10e6,type=int)
    parser.add_argument('--show_img', default='F')
    parser.add_argument('--train_prop', default=0.8,type=float)
    parser.add_argument('--epochs', default=50,type=int)
    parser.add_argument('--augment', default='None')

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data

    if (args.augment=='None'):
	print('No augmentation')
	start = time.time()
	train_X,train_Y,test_X,test_Y,class_train,class_test=augment_none(args)
	loss,val_loss,epochs= train(train_X,train_Y,test_X,test_Y)
	precision_sfg,recall_sfg,F1_score_sfg,accuracy_sfg,precision_ss,recall_ss,F1_score_ss,accuracy_ss,precision_fs,recall_fs,F1_score_fs,accuracy_fs, precision_all,recall_all,F1_score_all,accuracy_all,sfg_tp,sfg_fp,sfg_tn,sfg_fn,ss_tp,ss_fp,ss_tn,ss_fn,fs_tp,fs_fp,fs_tn,fs_fn,all_tp,all_fp,all_tn,all_fn,df_all= test(test_X,test_Y,class_test)
        data=np.array([precision_sfg,recall_sfg,F1_score_sfg,accuracy_sfg,precision_ss,recall_ss,F1_score_ss,accuracy_ss,precision_fs,recall_fs,F1_score_fs,accuracy_fs, precision_all,recall_all,F1_score_all,accuracy_all])*100
	end = time.time()
	print(end-start)
	print(sfg_tp,sfg_fp,sfg_tn,sfg_fn,ss_tp,ss_fp,ss_tn,ss_fn,fs_tp,fs_fp,fs_tn,fs_fn,all_tp,all_fp,all_tn,all_fn)
        np.savetxt(args.save_dir+str(args.snr)+'_'+str(args.freq)+'_none.txt', data,fmt='%f')

    if (args.augment=='Extended'):
	print('Augmenting extended sources')
	start = time.time()
 	train_X,train_Y,test_X,test_Y,class_train,class_test=augment(args)
	loss,val_loss,epochs= train(train_X,train_Y,test_X,test_Y)
	precision_sfg,recall_sfg,F1_score_sfg,accuracy_sfg,precision_ss,recall_ss,F1_score_ss,accuracy_ss,precision_fs,recall_fs,F1_score_fs,accuracy_fs, precision_all,recall_all,F1_score_all,accuracy_all,sfg_tp,sfg_fp,sfg_tn,sfg_fn,ss_tp,ss_fp,ss_tn,ss_fn,fs_tp,fs_fp,fs_tn,fs_fn,all_tp,all_fp,all_tn,all_fn,df_all= test(test_X,test_Y,class_test)
	end = time.time()
	print(end-start)
        data=np.array([precision_sfg,recall_sfg,F1_score_sfg,accuracy_sfg,precision_ss,recall_ss,F1_score_ss,accuracy_ss,precision_fs,recall_fs,F1_score_fs,accuracy_fs, precision_all,recall_all,F1_score_all,accuracy_all])*100
	print(sfg_tp,sfg_fp,sfg_tn,sfg_fn,ss_tp,ss_fp,ss_tn,ss_fn,fs_tp,fs_fp,fs_tn,fs_fn,all_tp,all_fp,all_tn,all_fn)
        np.savetxt(args.save_dir+str(args.snr)+'_'+str(args.freq)+'_extended.txt', data,fmt='%f')

    if (args.augment=='All'):
	start = time.time()
	print('Augmenting all sources')
 	train_X,train_Y,test_X,test_Y,class_train,class_test=augment(args)
	loss,val_loss,epochs= train(train_X,train_Y,test_X,test_Y)
	precision_sfg,recall_sfg,F1_score_sfg,accuracy_sfg,precision_ss,recall_ss,F1_score_ss,accuracy_ss,precision_fs,recall_fs,F1_score_fs,accuracy_fs, precision_all,recall_all,F1_score_all,accuracy_all,sfg_tp,sfg_fp,sfg_tn,sfg_fn,ss_tp,ss_fp,ss_tn,ss_fn,fs_tp,fs_fp,fs_tn,fs_fn,all_tp,all_fp,all_tn,all_fn,df_all= test(test_X,test_Y,class_test)
	end=time.time()
	print(end-start)
        data=np.array([precision_sfg,recall_sfg,F1_score_sfg,accuracy_sfg,precision_ss,recall_ss,F1_score_ss,accuracy_ss,precision_fs,recall_fs,F1_score_fs,accuracy_fs, precision_all,recall_all,F1_score_all,accuracy_all])*100
	print(sfg_tp,sfg_fp,sfg_tn,sfg_fn,ss_tp,ss_fp,ss_tn,ss_fn,fs_tp,fs_fp,fs_tn,fs_fn,all_tp,all_fp,all_tn,all_fn)
        np.savetxt(args.save_dir+str(args.snr)+'_'+str(args.freq)+'_all.txt', data,fmt='%f')





