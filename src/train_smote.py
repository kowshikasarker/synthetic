import argparse
from math import ceil
from pathlib import Path
from imblearn.over_sampling import SMOTE
import os
import sys
import pandas as pd
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
    parser.add_argument("--syn_sample_count", type=int,
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
                            syn_sample_count
                           ):
    train_mic = pd.read_csv(train_mic_path, sep='\t', index_col='Sample')
    train_met = pd.read_csv(train_met_path, sep='\t', index_col='Sample')
    train_meta = pd.read_csv(train_meta_path, sep='\t', index_col='Sample')
    
    print('train_meta', train_meta.columns)
    
    dummy_meta = train_meta.copy()
    dummy_meta.columns = list(range(dummy_meta.shape[1]))
    dummy_meta = pd.from_dummies(dummy_meta)
    meta_col = 'Cohort'
    dummy_meta.columns = [meta_col]
    print('dummy_meta', type(dummy_meta[meta_col]))
    
    
    cohort_pct = dict(dummy_meta[meta_col].value_counts(normalize=True))
    train_cohort_count = dict(dummy_meta[meta_col].value_counts(normalize=False))
    syn_cohort_count = {
        key: int(ceil((syn_sample_count*value)))+train_cohort_count[key] for key, value in cohort_pct.items()
    }
    
    print('cohort_pct', cohort_pct)
    print('train_cohort_count', train_cohort_count)
    print('syn_cohort_count', syn_cohort_count)
    
    train_df = pd.concat([train_mic, train_met], axis=1)
    train_df = pd.concat([train_df, dummy_meta], axis=1)
    
    train_y = train_df[meta_col].to_numpy().flatten()
    print('train_y', train_y)
    train_x = train_df.drop(columns=meta_col).to_numpy()
    
    print('train_mic', train_mic.shape,
          'train_met', train_met.shape,
          'train_meta', train_meta.shape,
          'train_x', train_x.shape,
          'train_y', train_y.shape
         )
    
    smote = SMOTE(sampling_strategy=syn_cohort_count)
    smote_x, smote_y = smote.fit_resample(train_x, train_y)
    
    smote_data = np.concatenate([smote_x, smote_y.reshape(-1, 1)], axis=1)
    smote_df = pd.DataFrame(smote_data,
                            columns=train_df.columns)
    print('train_df', smote_df.columns)
    print('smote_df', smote_df.columns)
    merged_df = pd.merge(smote_df,
                      train_df,
                      indicator=True,
                      how='left',
                      on=list(train_df.columns))
    syn_df = merged_df.query('_merge=="left_only"').drop('_merge', axis=1)
    rec_df = merged_df.query('_merge!="left_only"').drop('_merge', axis=1)
    rec_df.index = train_df.index
    
    rec_mic = rec_df[train_mic.columns]
    rec_met = rec_df[train_met.columns]
    rec_meta = rec_df[meta_col]
    
    rec_mic.to_csv('reconstructed_microbiome.tsv', sep='\t', index=True)
    rec_met.to_csv('reconstructed_metabolome.tsv', sep='\t', index=True)
    rec_meta.to_csv('reconstructed_metadata.tsv', sep='\t', index=True)
    
    syn_df.index = ['sample-'+str(i) for i in range(1, syn_df.shape[0]+1)]
    syn_df.index.name = 'Sample'
    
    syn_mic = syn_df[train_mic.columns]
    syn_met = syn_df[train_met.columns]
    syn_meta = syn_df[meta_col].astype(int)
    print('syn_meta', syn_meta.head())
    print('train_meta', list(train_meta.columns))
    syn_meta = pd.get_dummies(syn_meta).astype(int)
    syn_meta.columns = list(train_meta.columns)
    print('syn_meta', syn_meta.head())
    
    syn_mic.to_csv('synthetic_microbiome.tsv', sep='\t', index=True)
    syn_met.to_csv('synthetic_metabolome.tsv', sep='\t', index=True)
    syn_meta.to_csv('synthetic_metadata.tsv', sep='\t', index=True)
    
def main(args):
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(args.out_dir)
    
    log_file = open(args.out_dir + '/train_smote.log', 'w')
    
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