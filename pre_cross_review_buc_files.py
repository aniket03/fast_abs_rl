import os

import numpy as np
import pandas as pd

if __name__ == '__main__':

    # Set initial random seed and get DATA variable from environment
    seed_val = 3
    DATA_DIR = os.environ['DATA']
    np.random.seed(seed_val)

    # Get list of all train data files
    train_data_dir = os.path.join(DATA_DIR, 'train')
    all_train_files = os.listdir(train_data_dir)
    np.random.shuffle(all_train_files)

    # Get n_samples per bucket
    n_samples = len(all_train_files)
    n_buckets = 10
    n_samples_per_buc = int(n_samples / n_buckets)

    # Build dictionary bucket_samples_map
    bucket_samples_map = {}
    for bucket_id in range(n_buckets):
        st_ind = bucket_id * n_samples_per_buc
        en_ind = (bucket_id + 1) * n_samples_per_buc
        samples_in_curr_buc = []
        for sample_ind in range(st_ind, en_ind):
            sample_name = all_train_files[sample_ind]
            samples_in_curr_buc.append(sample_name)

        bucket_samples_map[bucket_id] = samples_in_curr_buc

    # Check that each bucket has unique elements
    for buc_i in range(n_buckets):
        buc_i_samples = set(bucket_samples_map[buc_i])
        for buc_j in range(buc_i+1, n_buckets):
            buc_j_samples = set(bucket_samples_map[buc_j])
            intersection_samples = buc_i_samples.intersection(buc_j_samples)
            if len(intersection_samples) > 0:
                print ('Error: one or more bucket has overlap')

    # Save a csv file cross_rev_buc_ind.csv
    for bucket_id in range(n_buckets):
        bucket_file_path = os.path.join(DATA_DIR, 'cross_rev_bucket_{}.csv'.format(str(bucket_id)))
        df = pd.DataFrame()
        df['filename'] = bucket_samples_map[bucket_id]
        df.to_csv(bucket_file_path)
