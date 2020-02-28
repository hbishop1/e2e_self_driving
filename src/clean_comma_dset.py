import os
import sys
import h5py
import glob
import numpy as np

root_dir = '/home2/pwkw48/4th_year_project/comma_dataset/valid'

previous_edited_files = glob.glob(root_dir + '/*/*edited.h5')

for f in previous_edited_files:
    print(f'Removing file {f}')
    os.remove(f)

image_files = sorted([os.path.join(root_dir,'camera', i) for i in os.listdir(os.path.join(root_dir,'camera'))])
log_files = sorted([os.path.join(root_dir,'log',i) for i in os.listdir(os.path.join(root_dir,'log'))])

for i in range(len(image_files)):
    with h5py.File(image_files[i],'r') as source_images:
        with h5py.File(log_files[i],'r') as source_log:   
            
            print(f'Extracting: {image_files[i]}')
            log_keys = ['speed_abs','car_accel', 'steering_angle']
            dest_length = min(int(source_images['X'].len() // 2), int(source_log['steering_angle'].len() // 10))
            with h5py.File(image_files[i][:-3]+'-edited.h5', "w") as dest_images:
                with h5py.File(log_files[i][:-3]+'-edited.h5', "w") as dest_log:

                    dest_images.create_dataset('image',(dest_length,3,160,320),dtype=np.uint8)
                    for k in log_keys:
                        dest_log.create_dataset(k,(dest_length,*source_log[k].shape[1:]),dtype=source_log[k].dtype)

                    for j in range(dest_length):
                        dest_images['image'][j] = source_images['X'][j*2]
                        for k in log_keys:
                            dest_log[k][j] = source_log[k][j*10]
            
        print('Completed')


