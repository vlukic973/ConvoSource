"""
This script generates the segmented real maps and solutions at a chosen exposure time, frequency and SNR on the simulated SKA data. The script command as it is generates 50x50 pixel images that are each spaced 50 pixels apart. The following commands segment the 560MHz at 8h exposure time dataset, at an SNR of 1. Run this first before 'source_finding_DNN_Bx_yh_v3.py'.

Usage:
       ... ...
python generate_real_data_and_solutions_Bx_yh_v1.py --generate_real_data 'F' --generate_solutions 'T' --img_inc 50 --img_size 50 --freq 1 --bg_fits '/path/to/bg_fits/SKAMid_B1_8h_pbf_corrected_v3.pybdsm.rmsd_I.fits' --fits_pbc '/path/to/primary_beam_corrected_image/SKAMid_B1_8h_pbf_corrected_v3.fits' --training_set '/path/to/training_set/TrainingSet_B1_v2.txt' --save_dir '/path/where/to/save/maps/' --snr 1

python generate_real_data_and_solutions_Bx_yh_v1.py --generate_real_data 'T' --generate_solutions 'F' --img_inc 50 --img_size 50 --freq 1 --bg_fits '/path/to/bg_fits/SKAMid_B1_8h_pbf_corrected_v3.pybdsm.rmsd_I.fits' --fits_pbc '/path/to/primary_beam_corrected_image/SKAMid_B1_8h_pbf_corrected_v3.fits' --training_set '/path/to/training_set/TrainingSet_B1_v2.txt' --save_dir '/path/where/to/save/maps/' --snr 1

           
Author: Vesna Lukic, E-mail: `vlukic973@gmail.com`
"""

from astropy.io import fits
import pandas as pd

import astropy
from astropy import coordinates
import astropy.units as u

import numpy as np

import matplotlib.pyplot as plt

import bdsf

def generate_data(args):
    """
Generating data
    """

    hdu1 = fits.open(args.fits_pbc)

    TrainingSet=pd.read_csv(args.training_set,skiprows=17,delimiter='\s+')

    TrainingSet=TrainingSet[TrainingSet.columns[0:15]]

    TrainingSet.columns=['ID','RA (core)','DEC (core)','RA (centroid)','DEC (centroid)','FLUX','Core frac','BMAJ','BMIN','PA','SIZE','CLASS','SELECTION','x','y']

    TrainingSet=TrainingSet[TrainingSet['SELECTION']==1]

    fits_image_filename = args.bg_fits

    hdul = fits.open(fits_image_filename)

    if (args.freq==1):

	    data1=hdul[0].data[0,0,16300:20300,16300:20300]
	    print('frequency is 560MHz')

    if (args.freq==2):
    
	    data1=hdul[0].data[0,0,16300:20500,16300:20500]
	    print('frequency is 1400MHz')

    if (args.freq==3):

	    data1=hdul[0].data[0,0,21700:25700,16300:20300]
	    print('frequency is 9200MHz')

    data1[np.isnan(data1)] = 0

    TrainingSet=TrainingSet[TrainingSet['FLUX']>args.snr*np.mean(data1)]

    print('Cut-off threshold is:',args.snr*np.mean(data1))

    TrainingSet['x']=np.round(TrainingSet['x'])
    TrainingSet['y']=np.round(TrainingSet['y'])

    TrainingSet['x']=TrainingSet['x'].astype(int)
    TrainingSet['y']=TrainingSet['y'].astype(int)

    TrainingSet=TrainingSet.reset_index(drop=True)

    if (args.freq==1):

	    TrainingSet['x']=TrainingSet['x']-16300
	    TrainingSet['y']=TrainingSet['y']-16300
            real_data1=hdu1[0].data[0,0,16300:20300,16300:20300]

    if (args.freq==2):
    	    print('resetting Training indices') 	
	    TrainingSet['x']=TrainingSet['x']-16300
	    TrainingSet['y']=TrainingSet['y']-16300
            real_data1=hdu1[0].data[0,0,16300:20500,16300:20500]

    if (args.freq==3):

	    TrainingSet['x']=TrainingSet['x']-16300
	    TrainingSet['y']=TrainingSet['y']-21700
            real_data1=hdu1[0].data[0,0,21700:25700,16300:20300]

    if ((args.img_inc==20) & (args.freq==1)):
	    cutout_size=3950
	    data=np.zeros((4000,4000,1), dtype=np.uint8 )	
    if ((args.img_inc==50) & (args.freq==1)):
	    cutout_size=4000
	    data=np.zeros((4000,4000,1), dtype=np.uint8 )	
    if ((args.img_inc==20) & (args.freq==2)):
	    cutout_size=4150
	    data=np.zeros((4200,4200,1), dtype=np.uint8 )	
    if ((args.img_inc==50) & (args.freq==2)):
	    cutout_size=4200
	    data=np.zeros((4200,4200,1), dtype=np.uint8 )	
    if ((args.img_inc==20) & (args.freq==3)):
	    cutout_size=3950
	    data=np.zeros((4000,4000,1), dtype=np.uint8 )	
    if ((args.img_inc==50) & (args.freq==3)):
	    cutout_size=4000
	    data=np.zeros((4000,4000,1), dtype=np.uint8 )

    print(cutout_size)
    for i in range(0,len(TrainingSet)):
	    print(i)
	    data[int(TrainingSet['y'][i:i+1]),int(TrainingSet['x'][i:i+1])] = TrainingSet['CLASS'][i:i+1]

    data=data.reshape(data.shape[0],data.shape[1])

    a=-1
    b=-1

    result_array = np.empty((0,args.img_size,args.img_size))

    if (args.generate_real_data=='T'):
	    print('generating real data')

	    for i in range(0,cutout_size,args.img_inc):
		a+=1
	 	for j in range(0,cutout_size,args.img_inc):
			b+=1
			if (b==int(cutout_size/args.img_inc)):
				b=0
				print(a,b)
				result_array = np.append(result_array, [real_data1[i+0:i+args.img_size,j+0:j+args.img_size]], axis=0)
			else:
				print(a,b)
				result_array = np.append(result_array, [real_data1[i+0:i+args.img_size,j+0:j+args.img_size]], axis=0)
	
	    np.save(args.save_dir+'real_images_'+str(args.img_size)+'_'+str(args.img_inc)+'_'+str(args.snr)+'_final.npy',result_array)

    else:

	    print('not generating real data')

    a=-1
    b=-1

    result_array = np.empty((0,args.img_size,args.img_size))

    if (args.generate_solutions=='T'):
	    print('generating solutions')

	    for i in range(0,cutout_size,args.img_inc):
		a+=1
	 	for j in range(0,cutout_size,args.img_inc):
			b+=1
			if (b==int(cutout_size/args.img_inc)):
				b=0
				print(a,b)
				result_array = np.append(result_array, [data[i+0:i+args.img_size,j+0:j+args.img_size]], axis=0)
			else:
				print(a,b)
				result_array = np.append(result_array, [data[i+0:i+args.img_size,j+0:j+args.img_size]], axis=0)
	
	    np.save(args.save_dir+'solutions_'+str(args.img_size)+'_'+str(args.img_inc)+'_'+str(args.snr)+'_final.npy',result_array)

    else:

	    print('not generating solutions')


    return TrainingSet


if __name__ == "__main__":
    import os
    import argparse
    from keras import callbacks

    parser = argparse.ArgumentParser(description="Generate data for source-finding")
    parser.add_argument('--fits_pbc', default='/path/to/primary_beam_corrected_image/SKAMid_B1_8h_pbf_corrected_v3.fits')
    parser.add_argument('--training_set', default='/path/to/training_set/TrainingSet_B1_v2.txt')
    parser.add_argument('--save_dir', default='/path/where/to/save/maps/')
    parser.add_argument('--bg_fits', default='/path/to/bg_fits/SKAMid_B1_8h_pbf_corrected_v3.pybdsm.rmsd_I.fits')
    parser.add_argument('--freq', default=1,type=int)
    parser.add_argument('--snr', default=5, type=int)
    parser.add_argument('--img_inc', default=50,type=int)
    parser.add_argument('--img_size', default=50,type=int)
    parser.add_argument('--generate_real_data', default='F')
    parser.add_argument('--generate_solutions', default='F')

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    Training_set= generate_data(args)
    print(Training_set[0:5])






