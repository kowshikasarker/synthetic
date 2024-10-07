import pandas as pd
import os
from itertools import product
from pathlib import Path
from subprocess import Popen
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_no", type=int,
                        help="index of the dataset to preprocess",
                        required=True, default=None)
parser.add_argument("--out_path", type=str,
                        help="path to the .sh file to write the commands",
                        required=True, default=None)
args = parser.parse_args()


base_data_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Data/microbiome-metabolome'
base_result_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Result/PRADA'

script_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Code/'
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

missing_pct = [0.20] 
corr_top_k = [3, 5]
prior_top_k = [3, 5]
train_pct = [0.50]
val_pct = [0.10]
model_name = ['separate_hidden', 'combined_hidden']
optim_loss = ['mse kl bce', 'mse kl']
prior = ['prior', 'no-prior']

Path(base_result_dir).mkdir(parents=True, exist_ok=True)

configs = list(product(missing_pct,
                        corr_top_k,
                        prior_top_k,
                        train_pct,
                        val_pct,
                        model_name,
                        optim_loss,
                        prior))

config_count = len(configs)
config_labels = ['config-'+str(i) for i in range(1, config_count+1)]

config_df = pd.DataFrame(configs,
                         columns=['missing_pct',
                            'corr_top_k',
                            'prior_top_k',
                            'train_pct',
                            'val_pct',
                            'model_name',
                            'optim_loss',
                            'prior'],
                         index=config_labels)

config_df.index.name = 'config_label'

config_df.to_csv(base_result_dir + '/config.tsv',
                 sep='\t',
                 index=True)

dataset = datasets[args.dataset_no]
print(dataset, end='\n', flush=True)
dataset_dir = base_data_dir + '/' + dataset

fp = open(args.out_path, 'w')

for config_no in range(config_count):
    config_label = config_labels[config_no]
    print(config_label, flush=True)
    config = configs[config_no]

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
    ' --' + config[7] + \
    ' --mic_mic_prior_path ' + mic_mic_prior_path + \
    ' --met_met_prior_path ' + met_met_prior_path + \
    ' --mic_met_prior_path ' + mic_met_prior_path + \
    ' --prior_top_k ' + str(config[2]) + \
    ' --out_dir ' + base_result_dir + '/' + dataset + '/' + config_label + '/preprocess'

    fp.write(command + ' & \n\n')
    
fp.flush()
fp.close()

os.system('chmod +x ' + args.out_path)
        
    
        
        