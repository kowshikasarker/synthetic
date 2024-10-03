import argparse
import os
import pandas as pd
from pathlib import Path
import sys
from subprocess import Popen

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mic_path", type=str,
                        help="path to the micriobiome data in .tsv format",
                        required=True, default=None)
    parser.add_argument("--met_path", type=str,
                        help="path to the metabolome data in .tsv format",
                        required=True, default=None)
    parser.add_argument("--meta_path", type=str,
                        help="path to the metadata data in .tsv format",
                        required=True, default=None)
    
    parser.add_argument("--missing_pct", type=str,
                        help="max percentage of missing values to keep microbiome or metabolome columns",
                        required=True, default=None)
    parser.add_argument("--imputation", type=str,
                        choices=['knn'],
                        help="method to impute values",
                        required=True, default=None)
    
    parser.add_argument('--corr', action=argparse.BooleanOptionalAction,
                       required=True, default=None)
    parser.add_argument("--corr_method", type=str, choices=['spearman', 'pearson'],
                        help="correlation method to contruct the correlation network",
                        required=False, default='spearman')
    parser.add_argument("--corr_top_k", type=str,
                        help="top k edges to consider in the correlation network",
                        required=False, default=None)
    
    parser.add_argument('--prior', action=argparse.BooleanOptionalAction,
                       required=True, default=None)
    parser.add_argument("--mic_mic_prior_path", type=str,
                        help="path to microbiome-microbiome prior edges in .tsv format",
                        required=False, default=None)
    parser.add_argument("--met_met_prior_path", type=str,
                        help="path to metabolome-metabolome prior edges in .tsv format",
                        required=False, default=None)
    parser.add_argument("--mic_met_prior_path", type=str,
                        help="path to microbiome-metabolome prior edges in .tsv format",
                        required=False, default=None)
    parser.add_argument("--prior_top_k", type=str,
                        help="top k edges to consider in the prior network",
                        required=False, default=None)
    
    parser.add_argument("--train_pct", type=str,
                        help="percentage of sample for training data",
                        required=True, default=None)
    parser.add_argument("--val_pct", type=str,
                        help="percentage of sample for validation data",
                        required=True, default=None)
    
    parser.add_argument("--model_name", type=str,
                        choices=['separate_hidden', 'combined_hidden', 'unconditional'],
                        help="name of the synthetic data generation model",
                        required=True, default=None)
    parser.add_argument("--optim_loss", type=str, nargs="+",
                        choices=['kl', 'mse', 'bce'],
                        help="losses to use for optimizing model parameters",
                        required=True, default=None)
    parser.add_argument("--hparam_loss", type=str,
                        choices=['kl', 'mse', 'bce'],
                        help="loss to use for tuning hyperparameters",
                        required=True, default=None)
    
    parser.add_argument("--runs", type=int,
                        help="no. of runs for each setting",
                        required=True, default=None)
    parser.add_argument("--out_dir", type=str,
                        help="path to the output dir",
                        required=True, default=None)
    args = parser.parse_args()
    return args

def pipeline(**kwargs):
    code_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Code/'
    preprocess_script = code_dir + '/preprocess.py'
    train_script = code_dir + '/train.py'
    evaluate_script = code_dir + '/evaluate.py'
    
    out_dir = os.getcwd()
    print('out_dir', out_dir)
    
    kwargs['out_dir'] = out_dir + '/preprocess'
        
    args = ['mic_path', 'met_path', 'meta_path',
            'missing_pct', 'imputation',
            'train_pct', 'val_pct', 'out_dir']
    corr_args = ['corr_method', 'corr_top_k']
    prior_args = ['mic_mic_prior_path', 'met_met_prior_path', 'mic_met_prior_path', 'prior_top_k']
    
    command = 'python3 ' + preprocess_script
    for arg in args:
        command = command + ' --' + arg + ' ' + kwargs[arg]
    
    if(kwargs['corr']):
        command = command + ' --corr'
        for arg in corr_args:
            command = command + ' --' + arg + ' ' + kwargs[arg]
    else:
        command = command + ' --no-corr'
        
    if(kwargs['prior']):
        command = command + ' --prior'
        for arg in prior_args:
            command = command + ' --' + arg + ' ' + kwargs[arg]
    else:
        command = command + ' --no-prior'
    
    print(command, end='\n')
    exit_code = os.system(command)
    if (exit_code != 0):
        print("ERROR", 'exit_code', exit_code, out_dir, end='\n')
    
    
    test_meta_df = pd.read_csv(out_dir + '/preprocess/test_metadata.tsv', sep='\t')
    
    kwargs['train_mic_path'] = out_dir + '/preprocess/train_microbiome.tsv'
    kwargs['train_met_path'] = out_dir + '/preprocess/train_metabolome.tsv'
    kwargs['train_meta_path'] = out_dir + '/preprocess/train_metadata.tsv'
    kwargs['val_mic_path'] = out_dir + '/preprocess/val_microbiome.tsv'
    kwargs['val_met_path'] = out_dir + '/preprocess/val_metabolome.tsv'
    kwargs['val_meta_path'] = out_dir + '/preprocess/val_metadata.tsv'
    kwargs['test_mic_path'] = out_dir + '/preprocess/test_microbiome.tsv'
    kwargs['test_met_path'] = out_dir + '/preprocess/test_metabolome.tsv'
    kwargs['test_meta_path'] = out_dir + '/preprocess/test_metadata.tsv'
    kwargs['edge_path'] = out_dir + '/preprocess/edges.tsv'
    kwargs['syn_sample_count'] = str(test_meta_df.shape[0])
    kwargs['optim_loss'] = ' '.join(kwargs['optim_loss'])
    
    args = ['train_mic_path', 'train_met_path', 'train_meta_path',
            'val_mic_path', 'val_met_path', 'val_meta_path',
            'edge_path', 'model_name', 'syn_sample_count',
            'optim_loss', 'hparam_loss', 'out_dir']
    
    for run in range(1, kwargs['runs']+1):
        kwargs['out_dir'] = out_dir + '/run-' + str(run) + '/output'
        command = 'python3 ' + train_script
        for arg in args:
            print(arg)
            command = command + ' --' + arg + ' ' + kwargs[arg]
        print(command, end='\n')
        #Popen(command)
        exit_code = os.system(command)
        if (exit_code != 0):
            print("ERROR", 'exit_code', exit_code, out_dir, end='\n')
    
        
    '''args = ['train_mic_path', 'train_met_path', 'train_meta_path',
            'test_mic_path', 'test_met_path', 'test_meta_path',
            'syn_mic_path', 'syn_met_path', 'syn_meta_path',
            'out_dir']
    
    for run in range(1, kwargs['runs']+1):
        fp = open(out_dir + '/run-' + str(run) + '/output/logs/csv_logs/best_hparam.txt', 'r')
        best_hparam = fp.readline().strip()
        print('run-'+str(run), best_hparam)
        kwargs['syn_mic_path'] = out_dir + '/run-' + str(run) + '/output/' + best_hparam + '/synthetic_microbiome.tsv'
        kwargs['syn_met_path'] = out_dir + '/run-' + str(run) + '/output/' + best_hparam + '/synthetic_metabolome.tsv'
        kwargs['syn_meta_path'] = out_dir + '/run-' + str(run) + '/output/' + best_hparam + '/synthetic_metadata.tsv'
        kwargs['out_dir'] = out_dir + '/run-' + str(run) + '/output/' + best_hparam + '/evaluation'
        command = 'python3 ' + evaluate_script
        for arg in args:
            print(arg)
            command = command + ' --' + arg + ' ' + kwargs[arg]
        print(command, end='\n\n')
        exit_code = os.system(command)
        if (exit_code != 0):
            print("ERROR", 'exit_code', exit_code, end='\n')'''

def main(args):
    print('pipeline.py')
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(args.out_dir)
    
    log_file = open(args.out_dir + '/pipeline.log', 'w')
    
    original_stdout = sys.stdout
    sys.stdout = log_file
    
    original_stderr = sys.stderr
    sys.stderr = log_file
    
    print('========== pipeline.py ==========', end='\n')

    print(args)
    kwargs = vars(args)
    del kwargs['out_dir']
    
    with open('setting.tsv', 'w') as fp:
        fp.write('missing_pct' + '\t' + str(kwargs['missing_pct']) + '\n')
        fp.write('corr_top_k' + '\t' + str(kwargs['corr_top_k']) + '\n')
        fp.write('prior_top_k' + '\t' + str(kwargs['prior_top_k']) + '\n')
        fp.write('train_pct' + '\t' + str(kwargs['train_pct']) + '\n')
        fp.write('val_pct' + '\t' + str(kwargs['val_pct']) + '\n')
        fp.write('model_name' + '\t' + str(kwargs['model_name']) + '\n')
        fp.write('optim_loss' + '\t' + str(kwargs['optim_loss']) + '\n')
        fp.write('corr' + '\t' + str(kwargs['corr']) + '\n')
        fp.write('prior' + '\t' + str(kwargs['prior']))
    
    
    
    pipeline(**kwargs)
    
    print()

    sys.stdout = original_stdout
    sys.stderr = original_stderr
    
    log_file.flush()
    log_file.close()

if __name__ == "__main__":
    main(parse_args())
