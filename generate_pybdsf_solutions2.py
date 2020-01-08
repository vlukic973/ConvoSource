"""
This script outputs the PyBDSF results in the same format as the AutoSource results, again assuming the images have size 50x50 pixels and are spaced 50 pixels apart. The following command gets the PyBDSF results on the 560MHz data at 8h exposure time, at an SNR of 1.

Usage:

python generate_pybdsf_solutions2.py --img_inc 50 --img_size 50 --freq 1 --bg_fits '/path/to/background_fits/SKAMid_B1_8h_pbf_corrected_v3.pybdsm.rmsd_I.fits' --save_dir '/path/where/to/save/results/' --snr 1 --pybdsf_table '/path/to/pybdsf/table/SKAMid_B1_8h_pbf_corrected_v3_table.csv' --load_solutions '/path/to/segmented/solutions/solutions_50_50_1_final.npy'


Author: Vesna Lukic, E-mail: `vlukic973@gmail.com`

"""

from __future__ import division
import pandas as pd
import numpy as np
from astropy.io import fits
from sklearn.metrics.pairwise import euclidean_distances
import operator
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage

def pybdsf_sols(args):

    PyBDSF_table=pd.read_csv(args.pybdsf_table,skiprows=5)

    PyBDSF_table[' Xposn']=np.round(PyBDSF_table[' Xposn'])
    PyBDSF_table[' Yposn']=np.round(PyBDSF_table[' Yposn'])

    if (args.freq==1):

	PyBDSF_table=PyBDSF_table[(PyBDSF_table[' Xposn']>16300) & (PyBDSF_table[' Xposn']<20300) & (PyBDSF_table[' Yposn']>16300) & (PyBDSF_table[' Yposn']<20300)]

	hdul = fits.open(args.bg_fits)

	data=hdul[0].data[0,0,16300:20300,16300:20300]

	data[np.isnan(data)] = 0

	PyBDSF_table=PyBDSF_table[PyBDSF_table[' Total_flux']>args.snr*np.mean(data)]

	PyBDSF_table=PyBDSF_table.reset_index(drop=True)

	PyBDSF_table[' Xposn']=PyBDSF_table[' Xposn']-16300
	PyBDSF_table[' Yposn']=PyBDSF_table[' Yposn']-16300

        data = np.zeros((4000,4000,1), dtype=np.uint8 )

        for i in range(0,len(PyBDSF_table)):
   		print(i)
		data[int(PyBDSF_table[' Yposn'][i:i+1]),int(PyBDSF_table[' Xposn'][i:i+1])] = [1]

        data=data.reshape(4000,4000)

        a=-1
        b=-1

        result_array = np.empty((0,args.img_size,args.img_size))

        for i in range(0,4000,args.img_inc):
	      	a+=1
	    	for j in range(0,4000,args.img_inc):
	    		b+=1
	    		if (b==int(4000.0/args.img_inc)):
	    			b=0
	    			print(a,b)
	    			result_array = np.append(result_array, [data[i+0:i+args.img_size,j+0:j+args.img_size]], axis=0)

	    		else:
	    			print(a,b)
	    			result_array = np.append(result_array, [data[i+0:i+args.img_size,j+0:j+args.img_size]], axis=0)

    if (args.freq==2):

	PyBDSF_table=PyBDSF_table[(PyBDSF_table[' Xposn']>16300) & (PyBDSF_table[' Xposn']<20500) & (PyBDSF_table[' Yposn']>16300) & (PyBDSF_table[' Yposn']<20500)]

	hdul = fits.open(args.bg_fits)

	data=hdul[0].data[0,0,16300:20500,16300:20500]

	data[np.isnan(data)] = 0

	PyBDSF_table=PyBDSF_table[PyBDSF_table[' Total_flux']>args.snr*np.mean(data)]

	print(len(PyBDSF_table))

	PyBDSF_table=PyBDSF_table.reset_index(drop=True)

	PyBDSF_table[' Xposn']=PyBDSF_table[' Xposn']-16300
	PyBDSF_table[' Yposn']=PyBDSF_table[' Yposn']-16300

        data = np.zeros((4200,4200,1), dtype=np.uint8 )

        for i in range(0,len(PyBDSF_table)):
		print(i)
		data[int(PyBDSF_table[' Yposn'][i:i+1]),int(PyBDSF_table[' Xposn'][i:i+1])] = [1]

        data=data.reshape(4200,4200)

        a=-1
        b=-1

        result_array = np.empty((0,args.img_size,args.img_size))

	for i in range(0,4200,args.img_inc):
	    	a+=1
	    	for j in range(0,4200,args.img_inc):
	    		b+=1
	    		if (b==int(4200.0/args.img_inc)):
	    			b=0
	    			print(a,b)
	    			result_array = np.append(result_array, [data[i+0:i+args.img_size,j+0:j+args.img_size]], axis=0)

	    		else:
	    			print(a,b)
	    			result_array = np.append(result_array, [data[i+0:i+args.img_size,j+0:j+args.img_size]], axis=0)

    if (args.freq==3):

	PyBDSF_table=PyBDSF_table[(PyBDSF_table[' Xposn']>16300) & (PyBDSF_table[' Xposn']<20300) & (PyBDSF_table[' Yposn']>21700) & (PyBDSF_table[' Yposn']<25700)]

	hdul = fits.open(args.bg_fits)

	data=hdul[0].data[0,0,21700:25700,16300:20300]
	data[np.isnan(data)] = 0

	PyBDSF_table=PyBDSF_table[PyBDSF_table[' Total_flux']>args.snr*np.mean(data)]

	PyBDSF_table=PyBDSF_table.reset_index(drop=True)

	PyBDSF_table[' Xposn']=PyBDSF_table[' Xposn']-16300
	PyBDSF_table[' Yposn']=PyBDSF_table[' Yposn']-21700

        data = np.zeros((4000,4000,1), dtype=np.uint8 )

        for i in range(0,len(PyBDSF_table)):
		print(i)
		data[int(PyBDSF_table[' Yposn'][i:i+1]),int(PyBDSF_table[' Xposn'][i:i+1])] = [1]

        data=data.reshape(4000,4000)

        a=-1
        b=-1

        result_array = np.empty((0,args.img_size,args.img_size))

        for i in range(0,4000,args.img_inc):
	    	a+=1
	    	for j in range(0,4000,args.img_inc):
	    		b+=1
	    		if (b==int(4000.0/args.img_inc)):
	    			b=0
	    			print(a,b)
	    			result_array = np.append(result_array, [data[i+0:i+args.img_size,j+0:j+args.img_size]], axis=0)

	    		else:
	    			print(a,b)
	    			result_array = np.append(result_array, [data[i+0:i+args.img_size,j+0:j+args.img_size]], axis=0)


    np.save(args.save_dir+'PyBDSF_solutions_'+str(args.freq)+'_'+str(args.img_size)+'_'+str(args.img_inc)+'.npy',result_array)

    return result_array

def test(args):
    """
Testing
    """
    class_image=np.load(args.load_solutions)

    data2=class_image

    data2=np.where(data2==2, 1, data2) 
    data2=np.where(data2==3, 1, data2) 

    solutions_orig=data2

    solutions_all=solutions_orig

    solutions_all=solutions_all.reshape(solutions_all.shape[0],solutions_all.shape[1],solutions_all.shape[2],1)
    class_image=class_image.reshape(class_image.shape[0],class_image.shape[1],class_image.shape[2],1)

    train_proportion=args.train_prop

    test_Y=solutions_all[int(train_proportion*len(solutions_all)):len(solutions_all)]
    class_test=class_image[int(train_proportion*len(solutions_all)):len(solutions_all)]

    print(test_Y.shape)
    print(class_test.shape)

    test_Y = test_Y.astype('float32')
    class_test = class_test.astype('float32')

    recon_image=np.load(args.save_dir+'PyBDSF_solutions_'+str(args.freq)+'_'+str(args.img_size)+'_'+str(args.img_inc)+'.npy')
    recon_image=recon_image[int(train_proportion*len(solutions_all)):len(solutions_all)]

    print(int(train_proportion*len(solutions_all)),len(solutions_all))

    recon_image=recon_image.reshape(recon_image.shape[0],recon_image.shape[1],recon_image.shape[2],1)
    print(recon_image.shape)

    recon_image = recon_image.astype('float32')

    if (args.show_img=='True'):
        print('Now showing images and detected features')
	plt.ion()
	for i in np.where(class_test==1)[0][0:10]:
	    	plt.subplot(1,2,1); plt.axis('off'); plt.imshow(class_test[i,:,:,0])
	    	plt.subplot(1,2,2); plt.axis('off'); plt.imshow(recon_image[i,:,:,0])
		plt.savefig(args.save_dir+'test_X_Y_recon_pybdsf'+str(i)+'.png')
		plt.close()

    return test_Y,class_test,recon_image,np.sum(test_Y),np.sum(recon_image)

def calculate_metrics(test_Y,class_test,recon_image):

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

	from sklearn.metrics.pairwise import euclidean_distances

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

	import pandas as pd

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
	
	import pandas as pd

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

	import pandas as pd

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

	import pandas as pd

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

    return precision_sfg,recall_sfg,F1_score_sfg,accuracy_sfg,precision_ss,recall_ss,F1_score_ss,accuracy_ss,precision_fs,recall_fs,F1_score_fs,accuracy_fs, precision_all,recall_all,F1_score_all,accuracy_all,np.sum(df_sfg['TP_sfg']),np.sum(df_sfg['FP_sfg']),np.sum(df_sfg['TN_sfg']),np.sum(df_sfg['FN_sfg']),np.sum(df_ss['TP_ss']),np.sum(df_ss['FP_ss']),np.sum(df_ss['TN_ss']),np.sum(df_ss['FN_ss']),np.sum(df_fs['TP_fs']),np.sum(df_fs['FP_fs']),np.sum(df_fs['TN_fs']),np.sum(df_fs['FN_fs']),np.sum(df_all['TP_all']),np.sum(df_all['FP_all']),np.sum(df_all['TN_all']),np.sum(df_all['FN_all']),df_all

if __name__ == "__main__":
    import os
    import argparse
    from keras import callbacks

    parser = argparse.ArgumentParser(description="Get metrics from the PyBDSF source-finder")
    parser.add_argument('--save_dir', default='/path/where/to/save/results/')
    parser.add_argument('--load_solutions', default='/path/to/segmented/solutions/solutions_50_505_final.npy')
    parser.add_argument('--bg_fits', default='/path/to/background_fits/SKAMid_B1_8h_pbf_corrected_v3.pybdsm.rmsd_I.fits')
    parser.add_argument('--pybdsf_table', default='/path/to/pybdsf/table/SKAMid_B1_8h_pb_corrected_v3_table.csv')
    parser.add_argument('--freq', default=1,type=int)
    parser.add_argument('--snr', default=5, type=int)
    parser.add_argument('--train_prop', default=0.8,type=float)
    parser.add_argument('--img_inc', default=50,type=int)
    parser.add_argument('--img_size', default=50,type=int)
    parser.add_argument('--show_img', default='F')

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    Training_set= pybdsf_sols(args)
    test_Y11,class_test11,recon_image11,sum_test_Y,sum_recon_image= test(args)
    print(test_Y11.shape,class_test11.shape,recon_image11.shape)
    data1=np.array([sum_test_Y,sum_recon_image])
    precision_sfg,recall_sfg,F1_score_sfg,accuracy_sfg,precision_ss,recall_ss,F1_score_ss,accuracy_ss,precision_fs,recall_fs,F1_score_fs,accuracy_fs, precision_all,recall_all,F1_score_all,accuracy_all,sfg_tp,sfg_fp,sfg_tn,sfg_fn,ss_tp,ss_fp,ss_tn,ss_fn,fs_tp,fs_fp,fs_tn,fs_fn,all_tp,all_fp,all_tn,all_fn,df_all=calculate_metrics(test_Y11,class_test11,recon_image11)
    print(sfg_tp,sfg_fp,sfg_tn,sfg_fn,ss_tp,ss_fp,ss_tn,ss_fn,fs_tp,fs_fp,fs_tn,fs_fn,all_tp,all_fp,all_tn,all_fn)
    data=np.array([precision_sfg,recall_sfg,F1_score_sfg,accuracy_sfg,precision_ss,recall_ss,F1_score_ss,accuracy_ss,precision_fs,recall_fs,F1_score_fs,accuracy_fs, precision_all,recall_all,F1_score_all,accuracy_all])*100
    np.savetxt(args.save_dir+str(args.snr)+'_'+str(args.freq)+'_pybdsf.txt', data,fmt='%f')
    np.savetxt(args.save_dir+str(args.snr)+'_'+str(args.freq)+'_sum_test_Y_recon_pybdsf.txt', data1,fmt='%f')

