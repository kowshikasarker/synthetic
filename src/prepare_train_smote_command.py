import argparse
import pandas as pd
import os
from pathlib import Path

base_result_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Manuscript/corr-all/SMOTE'
script_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Code/corr-all'
train_script = script_dir + '/train_smote.py'

datasets = ['ST001386']

command_dir = base_result_dir + '/Command'
Path(command_dir).mkdir(exist_ok=True, parents=True)
fp = open(command_dir + '/command.sh', 'w')

configs = ['config-1', 'config-13', 'config-25']

for dataset_no in range(len(datasets)):
    dataset = datasets[dataset_no]
    
    for config in configs:
        preprocess_dir = base_result_dir + '/' + dataset + '/' + config + '/preprocess'
        out_dir = base_result_dir + '/' + dataset + '/' + config + '/output'
        Path(out_dir).mkdir(exist_ok=True, parents=True)
        
        test_meta_df = pd.read_csv(preprocess_dir + '/test_condition.tsv', sep='\t')
        syn_sample_count = test_meta_df.shape[0]
        
        command = 'python3 ' + train_script + \
        ' --train_feature_path ' + preprocess_dir + '/train_feature.tsv' + \
        ' --train_condition_path ' + preprocess_dir + '/train_condition.tsv' + \
        ' --val_feature_path ' + preprocess_dir + '/val_feature.tsv' + \
        ' --val_condition_path ' + preprocess_dir + '/val_condition.tsv' + \
        ' --syn_sample_count ' + str(syn_sample_count) + \
        ' --out_dir ' + out_dir + \
        ' &> ' + out_dir + '/wrapper.log' + \
        ' &'
        fp.write(command + '\n\n')
fp.flush()
fp.close()



        

