import argparse
import pandas as pd
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

base_result_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Result/PRADA'
script_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Code'
train_script = script_dir + '/train_prada.py'
command_dir =  base_result_dir + '/command'

datasets = [
    'YACHIDA_CRC_2019',
    'iHMP_IBDMDB_2019',
    'FRANZOSA_IBD_2019',
    'ERAWIJANTARI_GASTRIC_CANCER_2020',
    'MARS_IBS_2020',
    'KOSTIC_INFANTS_DIABETES_2015',
    'JACOBS_IBD_FAMILIES_2016',
    'KIM_ADENOMAS_2020',
    'SINHA_CRC_2016'
]

config_path =  base_result_dir + '/config.tsv'
config_df = pd.read_csv(config_path, sep='\t', index_col='config_label')

for dataset in datasets:
    filepath = command_dir + '/Train_PRADA.' + dataset + '.sh'
    fp = open(filepath, 'w')
    print(dataset, end='\n', flush=True)
    
    for config_label in config_df.index:
        config = config_df.loc[config_label, :]
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

    os.system('chmod +x ' + filepath)








    


        

