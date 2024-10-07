import argparse
import pandas as pd
import os
from pathlib import Path
from math import ceil

base_result_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Result/CVAE'
script_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Code'
train_script = script_dir + '/train_cvae.py'

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
    
model_names = ['separate_hidden', 'combined_hidden']
command_dir = base_result_dir + '/Command'
Path(command_dir).mkdir(exist_ok=True, parents=True)

commands = []

for dataset_no in range(len(datasets)):
    dataset = datasets[dataset_no]
    
    preprocess_dir = base_result_dir + '/' + dataset + '/preprocess'
    for model_name in model_names:
        out_dir = base_result_dir + '/' + dataset + '/' + model_name + '/output'
        Path(out_dir).mkdir(exist_ok=True, parents=True)
        
        test_meta_df = pd.read_csv(preprocess_dir + '/test_metadata.tsv', sep='\t')
        syn_sample_count = test_meta_df.shape[0]
        
        
        command = 'python3 ' + train_script + \
        ' --train_mic_path ' + preprocess_dir + '/train_microbiome.tsv' + \
        ' --train_met_path ' + preprocess_dir + '/train_metabolome.tsv' + \
        ' --train_meta_path ' + preprocess_dir + '/train_metadata.tsv' + \
        ' --val_mic_path ' + preprocess_dir + '/val_microbiome.tsv' + \
        ' --val_met_path ' + preprocess_dir + '/val_metabolome.tsv' + \
        ' --val_meta_path ' + preprocess_dir + '/val_metadata.tsv' + \
        ' --model_name ' + model_name + \
        ' --syn_sample_count ' + str(syn_sample_count) + \
        ' --out_dir ' + out_dir + \
        ' &> ' + out_dir + '/wrapper.log' + \
        ' &'
        commands.append(command)
        
split_size = 8
split_count = int(ceil(len(commands) / split_size))
print('split_count', split_count)
for i in range(split_count):
    file_path = command_dir + '/command-' + str(i+1) + '.sh'
    fp = open(file_path, 'w')
    start_idx = i*split_size
    end_idx = min((i+1)*split_size, len(commands))
    split = commands[start_idx: end_idx]
    for command in split:
        fp.write(command + '\n\n')
    fp.flush()
    fp.close()
    os.system('chmod +x ' + file_path)

        

        




        

