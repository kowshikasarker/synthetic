import argparse
import pandas as pd
import os
from pathlib import Path

base_result_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Result/PRADA'
script_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Code'
train_script = script_dir + '/train.py'

#WANG_ESRD_2020 -> skipped, cvae training  issues
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

'''splits = {
    'timan1.1': (1, 20),
    'timan1.2': (21, 40),
    'timan108.1': (41, 50),
    'timan108.2': (51, 60),
    'timan107.1': (61, 66),
    'timan107.2': (67, 72)
}'''

splits = {
    'timan108.1': (1, 10),
    'timan108.2': (11, 20),
    'timan107.1': (21, 26),
    'timan107.2': (27, 32)
}

config_path =  base_result_dir + '/config.tsv'
config_df = pd.read_csv(config_path, sep='\t', index_col='config_label')

def prepare_commands(dataset, start_config, end_config, out_path):
    fp = open(out_path, 'w')
    
    for config_no in range(start_config, end_config+1):
        config_label = 'config-' + str(config_no)
        config = config_df.loc[config_label, :].to_numpy()
        
        preprocess_dir = base_result_dir + '/' + dataset + '/' + config_label + '/preprocess'
        out_dir = base_result_dir + '/' + dataset + '/' + config_label + '/output'
        Path(out_dir).mkdir(exist_ok=True, parents=True)
        
        test_meta_df = pd.read_csv(preprocess_dir + '/test_metadata.tsv', sep='\t')
        
        command = 'python3 ' + train_script + \
        ' --train_mic_path ' + preprocess_dir + '/train_microbiome.tsv' + \
        ' --train_met_path ' + preprocess_dir + '/train_metabolome.tsv' + \
        ' --train_meta_path ' + preprocess_dir + '/train_metadata.tsv' + \
        ' --val_mic_path ' + preprocess_dir + '/val_microbiome.tsv' + \
        ' --val_met_path ' + preprocess_dir + '/val_metabolome.tsv' + \
        ' --val_meta_path ' + preprocess_dir + '/val_metadata.tsv' + \
        ' --edge_path ' + preprocess_dir + '/edges.tsv' + \
        ' --model_name ' + config[5] + \
        ' --syn_sample_count ' + str(test_meta_df.shape[0]) + \
        ' --optim_loss ' + config[6] + \
        ' --hparam_loss mse' + \
        ' --out_dir ' + out_dir + \
        ' &> ' + out_dir + '/wrapper.log' + \
        ' &'
        fp.write(command + '\n\n')
    fp.flush()
    fp.close()

for dataset_no in range(len(datasets)):
    dataset = datasets[dataset_no]
    for key, value in splits.items():
        out_path = base_result_dir + '/Command/Train.D' + str(dataset_no) + '.' + key + '.sh'
        prepare_commands(dataset, value[0], value[1], out_path)
        os.system('chmod +x ' + out_path)








    


        

