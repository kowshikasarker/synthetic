import pandas as pd
import os
from itertools import product
from pathlib import Path
from subprocess import Popen
import warnings
warnings.filterwarnings("ignore")


base_data_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Data/microbiome-metabolome'
base_result_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Result/PRADA'

script_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Code'
preprocess_script = script_dir + '/preprocess.py'

mic_mic_prior_path = '/shared/nas/data/m1/ksarker2/Synthetic/Data/prior/GutNet.tsv'
met_met_prior_path = '/shared/nas/data/m1/ksarker2/Synthetic/Data/prior/Human1.tsv'
mic_met_prior_path = '/shared/nas/data/m1/ksarker2/Synthetic/Data/prior/GMMAD.tsv'

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

config_df = pd.read_csv(base_result_dir + '/config.tsv',
                        sep='\t',
                        index_col='config_label')

command_dir =  base_result_dir + '/command'
Path(command_dir).mkdir(exist_ok=True, parents=True)

for dataset in datasets:
    filepath = command_dir + '/Preprocess.' + dataset + '.sh'
    fp = open(filepath, 'w')
    print(dataset, end='\n', flush=True)
    dataset_dir = base_data_dir + '/' + dataset
    for config_label in config_df.index:
        config = config_df.loc[config_label, :]
    
        command = 'python3 ' + preprocess_script + \
        ' --mic_path ' + dataset_dir + '/microbiome.tsv' + \
        ' --met_path ' + dataset_dir + '/metabolome.tsv' + \
        ' --meta_path ' + dataset_dir + '/metadata.tsv' + \
        ' --missing_pct ' + str(config[0]) + \
        ' --imputation knn' + \
        ' --train_pct ' + str(config[3]) + \
        ' --val_pct ' + str(config[4]) + \
        ' --corr ' + \
        ' --corr_method spearman' + \
        ' --corr_top_k ' + str(config[1]) + \
        ' --' + config[8] + \
        ' --mic_mic_prior_path ' + mic_mic_prior_path + \
        ' --met_met_prior_path ' + met_met_prior_path + \
        ' --mic_met_prior_path ' + mic_met_prior_path + \
        ' --prior_top_k ' + str(config[2]) + \
        ' --out_dir ' + base_result_dir + '/' + dataset + '/' + config_label + '/preprocess'

        fp.write(command + ' & \n\n')
    
    fp.flush()
    fp.close()

    os.system('chmod +x ' + filepath)
        
    
        
        