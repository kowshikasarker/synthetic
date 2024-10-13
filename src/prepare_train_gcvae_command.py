import argparse
import pandas as pd
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

base_result_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Result/Phase-3/GCVAE'
train_script = '/shared/nas/data/m1/ksarker2/Synthetic/Code/Phase-3/train_gcvae.py'
command_dir =  base_result_dir + '/command'

'''datasets = [
    'YACHIDA_CRC_2019',
    'iHMP_IBDMDB_2019',
    'MARS_IBS_2020'
]'''

datasets = ['ST001386']

config_path =  base_result_dir + '/config.tsv'
config_df = pd.read_csv(config_path, sep='\t', index_col='config_label')

for dataset in datasets:
    filepath = command_dir + '/Train_GCVAE.' + dataset + '.sh'
    fp = open(filepath, 'w')
    print(dataset, end='\n', flush=True)
    
    for config_label in config_df.index:
        config = config_df.loc[config_label, :]
        preprocess_dir = base_result_dir + '/' + dataset + '/' + config_label + '/preprocess'
        out_dir = base_result_dir + '/' + dataset + '/' + config_label + '/output'
        Path(out_dir).mkdir(exist_ok=True, parents=True)
        
        syn_sample_count = pd.read_csv(preprocess_dir + '/test_condition.tsv', sep='\t').shape[0]
        
        command = 'python3 ' + train_script + \
        ' --train_feature_path ' + preprocess_dir + '/train_feature.tsv' + \
        ' --train_condition_path ' + preprocess_dir + '/train_condition.tsv' + \
        ' --val_feature_path ' + preprocess_dir + '/val_feature.tsv' + \
        ' --val_condition_path ' + preprocess_dir + '/val_condition.tsv' + \
        ' --edge_path ' + preprocess_dir + '/edges.tsv' + \
        ' --model_name ' + config[5] + \
        ' --syn_sample_count ' + str(syn_sample_count) + \
        ' --out_dir ' + out_dir + \
        ' &> ' + out_dir + '/wrapper.log' + \
        ' &'
        fp.write(command + '\n\n')
    
    fp.flush()
    fp.close()

    os.system('chmod +x ' + filepath)
