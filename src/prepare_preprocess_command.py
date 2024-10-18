import pandas as pd
import os
from itertools import product
from pathlib import Path
from subprocess import Popen
import warnings
warnings.filterwarnings("ignore")

base_data_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Data/metabolomics_workbench'
base_result_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Manuscript/corr-all/GCVAE'

script_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Code/corr-all'
preprocess_script = script_dir + '/preprocess.py'

met_prior_path = '/shared/nas/data/m1/ksarker2/Synthetic/Data/prior/KEGG.tsv'
met_filename = 'metabolome.tsv'

datasets = ['ST001386']

feature_count = [10, 40, 70, 100]
corr_top_pct = [0.10, 0.15, 0.20, 0.25, 0.30]
prior_top_pct = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]
missing_pct = [0.5]
train_pct = [0.60]
val_pct = [0.10]
model_name = ['combined_hidden']

configs = list(product(feature_count,
                       corr_top_pct,
                       prior_top_pct,
                       missing_pct,
                       train_pct,
                       val_pct,
                       model_name))

config_count = len(configs)
config_labels = ['config-'+str(i) for i in range(1, config_count+1)]
config_df = pd.DataFrame(configs,
                         columns=['feature_count',
                                  'corr_top_pct',
                                  'prior_top_pct',
                                  'missing_pct',
                                  'train_pct',
                                  'val_pct',
                                  'model_name'],
                         index=config_labels)
config_df.index.name = 'config_label'
config_df.to_csv(base_result_dir + '/config.tsv', sep='\t', index=True)

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
        ' --feature_path ' + dataset_dir + '/' + met_filename + \
        ' --condition_path ' + dataset_dir + '/metadata.tsv' + \
        ' --missing_pct ' + str(config[3]) + \
        ' --imputation knn' + \
        ' --feature_count ' + str(config[0]) + \
        ' --train_pct ' + str(config[4]) + \
        ' --val_pct ' + str(config[5]) + \
        ' --corr_top_pct ' + str(config[1]) + \
        ' --prior_path ' + met_prior_path + \
        ' --prior_top_pct ' + str(config[2]) + \
        ' --out_dir ' + base_result_dir + '/' + dataset + '/' + config_label + '/preprocess'
        fp.write(command + ' & \n\n')
    
    fp.flush()
    fp.close()
    os.system('chmod +x ' + filepath)
        
    
        
        