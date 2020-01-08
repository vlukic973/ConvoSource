Python code for source-finding in radio astronomical images

First go to the SDC1 webpage https://astronomers.skatelescope.org/ska-science-data-challenge-1/ and download the primary beams across the three frequencies (560, 1400 and 9200 MHz) under ‘Ancillary data’. Also download the 9 FITS files under ‘Data’ as well as the 3 ‘Training set’ files. The 9 data files are for 3 different exposure times (8h, 100h and 1000h) as well as the 3 different frequencies. There are only 3 training set files as they only depend on the frequency.

The first step is to correct for the primary beam, using the primary beam image and the original FITS files, which can be done using CASA. Ensure the final output file is in FITS format.

Next, use the PyBDSF ‘process_image’ command, using the option ‘rms_map=True’ to output the background noise map, which should end in ‘.pybdsm.rmsd_I.fits’, as well as the PyBDSF table of detected sources.

Ensure keras is installed before running any of the python scripts. Download the script ‘generate_real_data_and_solutions_Bx_yh_v1.py’. The purpose of this file is to generate the segmented maps of the real images and the ‘solution map’; the positions of the centroid of each source, for a given exposure time, frequency and SNR. Currently the command to run the script specifies 50x50 pixel images spaced 50 pixels apart. Eg for 8h exposure time at 560 MHz and using an SNR of 1:

The following command generates the segmented solutions:

python generate_real_data_and_solutions_Bx_yh_v1.py --generate_real_data 'F' --generate_solutions 'T' --img_inc 50 --img_size 50 --freq 1 --bg_fits '/lofar5/stvf319/SKA_v3/SKAMid_B1_8h_pbf_corrected_v3_pybdsm/04Mar2019_10.30.38/background/SKAMid_B1_8h_pbf_corrected_v3.pybdsm.rmsd_I.fits' --fits_pbc '/lofar5/stvf319/SKA_v3/SKAMid_B1_8h_pbf_corrected_v3.fits' --training_set '/lofar5/stvf319/SKA/TrainingSet_B1_v2.txt' --save_dir '/lofar5/stvf319/SKA_v5/B1_8h/snr_1/' --snr 1

To generate the real segmented maps:

python generate_real_data_and_solutions_Bx_yh_v1.py --generate_real_data 'T' --generate_solutions 'F' --img_inc 50 --img_size 50 --freq 1 --bg_fits '/lofar5/stvf319/SKA_v3/SKAMid_B1_8h_pbf_corrected_v3_pybdsm/04Mar2019_10.30.38/background/SKAMid_B1_8h_pbf_corrected_v3.pybdsm.rmsd_I.fits' --fits_pbc '/lofar5/stvf319/SKA_v3/SKAMid_B1_8h_pbf_corrected_v3.fits' --training_set '/lofar5/stvf319/SKA/TrainingSet_B1_v2.txt' --save_dir '/lofar5/stvf319/SKA_v5/B1_8h/snr_1/' --snr 1

The –bg_fits is the option for the background noise image, the –pbc is the primary beam corrected fits file, and the –training_set file is the file containing all the sources and their properties.

Download the script ‘source_finding_DNN_Bx_yh_v3.py’. This is the file that trains and tests AutoSource, assuming that the ‘generate_real_data_and_solutions_Bx_yh_v1.py’ script has been run first. The script again assumes there are 50x50 pixel images spaced 50 pixels apart. It is possible to augment different sets of images, and to run the training and testing parts separately. 

Example usage at 8h exposure time, 560MHz, SNR=1, no augmentation applied:

python source_finding_DNN_Bx_yh_v3.py --img_inc 50 --img_size 50 --freq 1 --bg_fits '/path/to/background_fits/SKAMid_B1_8h_pbf_corrected_v3.pybdsm.rmsd_I.fits' --save_dir '/path/where/to/save/results/' --augment 'None' --epochs 50 --load_images '/path/to/segmented/images/real_images_50_50_1_final.npy' --load_solutions '/path/to/segmented/solutions/solutions_50_50_1_final.npy' --snr 1 

Augmenting the ‘extended’ images only (ss and fs sources):

python source_finding_DNN_Bx_yh_v3.py --img_inc 50 --img_size 50 --freq 1 --bg_fits '/path/to/background_fits/SKAMid_B1_8h_pbf_corrected_v3.pybdsm.rmsd_I.fits' --save_dir '/path/where/to/save/results/' --augment 'Extended' --epochs 50 --load_images '/path/to/segmented/images/real_images_50_50_1_final.npy' --load_solutions '/path/to/segmented/solutions/solutions_50_50_1_final.npy' --snr 1

Augmenting ‘all’ images (SFGs, ss and fs sources)

python source_finding_DNN_Bx_yh_v3.py --img_inc 50 --img_size 50 --freq 1 --bg_fits '/path/to/background_fits/SKAMid_B1_8h_pbf_corrected_v3.pybdsm.rmsd_I.fits' --save_dir '/path/where/to/save/results/' --augment 'All' --epochs 50 --load_images '/path/to/segmented/images/real_images_50_50_1_final.npy' --load_solutions '/path/to/segmented/solutions/solutions_50_50_1_final.npy' --snr 1

Next, we can generate the equivalent metrics across the sources from the PyBDSF source-finder, for easy comparison to the AutoSource results. The following command gets the PyBDSF results on the 560MHz data at 8h exposure time, at an SNR of 1:

python generate_pybdsf_solutions2.py --img_inc 50 --img_size 50 --freq 1 --bg_fits '/path/to/background_fits/SKAMid_B1_8h_pbf_corrected_v3.pybdsm.rmsd_I.fits' --save_dir '/path/where/to/save/results/' --snr 1 --pybdsf_table '/path/to/pybdsf/table/SKAMid_B1_8h_pbf_corrected_v3_table.csv' --load_solutions '/path/to/segmented/solutions/solutions_50_50_1_final.npy'


