import os
import sys
import h5py
import glob
import numpy as np

root_dir = '/home2/pwkw48/4th_year_project/comma_dataset/'

previous_edited_files = glob.glob(root_dir + '/*/log/*edited.h5')

for f in previous_edited_files:
    print(f'Removing file {f}')
    os.remove(f)

log_files = glob.glob(root_dir + '/*/log/*.h5')

for f in log_files:
    with h5py.File(f,'r') as source_log:   
        
        print(f'Extracting: {f}')
        log_keys = ['speed_abs','car_accel', 'steering_angle']
        dest_length = int(max(source_log['cam1_ptr'][:])) + 1
        with h5py.File(f[:-3]+'-edited.h5', "w") as dest_log:

            dset = {}
            for k in log_keys:
                dset[k] = np.zeros((int(dest_length)))

            last_cam_ptr = -1
            for c,p in enumerate(source_log['cam1_ptr']):
                if p > last_cam_ptr:
                    last_cam_ptr = p
                    for k in log_keys:
                        dset[k][int(p)] = source_log[k][c]

            for k in log_keys:
                dest_log.create_dataset(k,data=dset[k])
            
        print('Completed')


