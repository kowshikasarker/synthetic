import argparse
from pathlib import Path
import pandas as pd
import os
import sys
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_mic_path", type=str,
                        help="path to the train micriobiome data in .tsv format",
                        required=True, default=None)
    parser.add_argument("--train_met_path", type=str,
                        help="path to the train metabolome data in .tsv format",
                        required=True, default=None)
    parser.add_argument("--train_meta_path", type=str,
                        help="path to the train metadata data in .tsv format",
                        required=True, default=None)
    parser.add_argument("--method", type=str,
                        choices=['normal', 'uniform'],
                        help="count of synthetic samples to generate",
                        required=True, default=None)
    parser.add_argument("--out_dir", type=str,
                        help="path to output dir",
                        required=True, default=None)
    args = parser.parse_args()
    return args



def generate_synthetic_data(train_mic_path,
                            train_met_path,
                            train_meta_path,
                            method
                           ):
    train_mic = pd.read_csv(train_mic_path, sep='\t', index_col='Sample')
    train_met = pd.read_csv(train_met_path, sep='\t', index_col='Sample')
    train_meta = pd.read_csv(train_meta_path, sep='\t', index_col='Sample')
    train_df = pd.concat([train_mic, train_met], axis=1)
    
    train_meta.index = ['sample-'+str(i) for i in range(1, train_meta.shape[0]+1)]
    train_meta.index.name = 'Sample'
    train_meta.to_csv('synthetic_metadata.tsv', sep='\t', index=True)
    
    noise = None
    
    if (method == 'normal'):
        noise = np.random.normal(0, 1, train_df.shape)
    elif (method == 'uniform'):
        noise = np.random.uniform(-1.0, 1.0, train_df.shape)
    else:
        raise Exception('Unrecognized method', method)
    noise_df = pd.DataFrame(noise,
                            columns=train_df.columns,
                            index=train_df.index)
    syn_df = train_df.add(noise_df)
    lower = train_df.min(axis=0)
    upper = train_df.max(axis=0)
    syn_df = syn_df.clip(lower, upper, axis=1)
    
    syn_df.index = ['sample-'+str(i) for i in range(1, syn_df.shape[0]+1)]
    syn_df.index.name = 'Sample'
    
    syn_mic = syn_df[train_mic.columns]
    syn_met = syn_df[train_met.columns]
    syn_mic.to_csv('synthetic_microbiome.tsv', sep='\t', index=True)
    syn_met.to_csv('synthetic_metabolome.tsv', sep='\t', index=True)
    
def main(args):
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(args.out_dir)
    
    log_file = open(args.out_dir + '/train_noise.log', 'w')
    
    original_stdout = sys.stdout
    sys.stdout = log_file
    
    original_stderr = sys.stderr
    sys.stderr = log_file
    
    print(args)
    
    kwargs = vars(args)
    del kwargs['out_dir']
    generate_synthetic_data(**kwargs)

    sys.stdout = original_stdout
    sys.stderr = original_stderr
    
    log_file.flush()
    log_file.close()


if __name__ == "__main__":
    main(parse_args())