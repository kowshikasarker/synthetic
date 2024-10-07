import argparse
import pandas as pd
import os
from pathlib import Path

base_result_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Result/NOISE'
script_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Code'
train_script = script_dir + '/train_noise.py'

datasets = [
        'YACHIDA_CRC_2019',
        'iHMP_IBDMDB_2019',
        'WANG_ESRD_2020',
        'FRANZOSA_IBD_2019',
        'ERAWIJANTARI_GASTRIC_CANCER_2020',
        'MARS_IBS_2020',
        'KOSTIC_INFANTS_DIABETES_2015',
        'JACOBS_IBD_FAMILIES_2016',
        'KIM_ADENOMAS_2020',
        'SINHA_CRC_2016',
        'HE_INFANTS_MFGM_2019',
        'KANG_AUTISM_2017'
    ]
    
methods = ['normal', 'uniform']
command_dir = base_result_dir + '/Command'
Path(command_dir).mkdir(exist_ok=True, parents=True)
fp = open(command_dir + '/command.sh', 'w')

for dataset_no in range(len(datasets)):
    dataset = datasets[dataset_no]
    
    preprocess_dir = base_result_dir + '/' + dataset + '/preprocess'
    for method in methods:
        out_dir = base_result_dir + '/' + dataset + '/' + method + '/output'
        Path(out_dir).mkdir(exist_ok=True, parents=True)
        
        test_meta_df = pd.read_csv(preprocess_dir + '/test_metadata.tsv', sep='\t')
        syn_sample_count = test_meta_df.shape[0]
        
        
        command = 'python3 ' + train_script + \
        ' --train_mic_path ' + preprocess_dir + '/train_microbiome.tsv' + \
        ' --train_met_path ' + preprocess_dir + '/train_metabolome.tsv' + \
        ' --train_meta_path ' + preprocess_dir + '/train_metadata.tsv' + \
        ' --method ' + method + \
        ' --out_dir ' + out_dir + \
        ' &> ' + out_dir + '/wrapper.log' + \
        ' &'
        fp.write(command + '\n\n')
fp.flush()
fp.close()



        

