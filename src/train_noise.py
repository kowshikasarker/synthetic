import argparse
from pathlib import Path
from imblearn.over_sampling import SMOTE
import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_feature_path", type=str,
                        required=True, default=None)
    parser.add_argument("--train_condition_path", type=str,
                        required=True, default=None)
    
    parser.add_argument("--val_feature_path", type=str,
                        required=True, default=None)
    parser.add_argument("--val_condition_path", type=str,
                        required=True, default=None)
    
    parser.add_argument("--syn_sample_count", type=int,
                        required=True, default=None)
    
    parser.add_argument("--out_dir", type=str,
                        help="path to output dir",
                        required=True, default=None)
    args = parser.parse_args()
    return args

def generate_synthetic_data(train_feature_path,
                            train_condition_path,
                            val_feature_path,
                            val_condition_path,
                            syn_sample_count):
    train_feature = pd.read_csv(train_feature_path, sep='\t', index_col='Sample')
    train_condition = pd.read_csv(train_condition_path, sep='\t', index_col='Sample')
    condition_cols = list(train_condition.columns)
    train_condition = pd.from_dummies(train_condition)
    print('train_condition', train_condition)
    
    val_feature = pd.read_csv(val_feature_path, sep='\t', index_col='Sample')
    val_condition = pd.read_csv(val_condition_path, sep='\t', index_col='Sample')
    val_condition = pd.from_dummies(val_condition)
    print('train_condition', val_condition)

    train_x = train_feature.to_numpy()
    train_y = train_condition.to_numpy().flatten()

    val_x = val_feature.to_numpy()
    val_y = val_condition.to_numpy().flatten()
    
    print('train_x', train_x.shape,
          'train_y', train_y.shape,
          'val_x', val_x.shape,
          'val_y', val_y.shape)
    
    assert train_x.shape[0] >= syn_sample_count
    
    train_x, _, train_y, _ = train_test_split(train_x, train_y, train_size=syn_sample_count / train_x.shape[0], stratify=train_y)
    
    print('train_x', train_x.shape,
          'train_y', train_y.shape)

    noise_distr = ['uniform', 'normal']
    hparam_label = ['hparam-'+str(i) for i in range(len(noise_distr))]
    
    hparam_df = pd.DataFrame(zip(hparam_label, noise_distr),
                             columns=['hparam_label', 'noise'])
    
    log_dir = os.getcwd() + '/logs'
    auroc = []
    for i in range(len(hparam_label)):
        hparam_out_dir = hparam_label[i]
        Path(os.getcwd() + '/' + hparam_out_dir).mkdir(exist_ok=True, parents=True)
        
        noise = None
        if(noise_distr[i] == 'uniform'):
            noise = np.random.uniform(-1.0, 1.0, train_x.shape)
        elif(noise_distr[i] == 'normal'):
            noise = np.random.normal(0, 1, train_x.shape)
        else:
            raise Exception('Unrecognized noise distribution', noise_distr[i])
            
        syn_x = np.add(train_x, noise)
        syn_y = train_y.copy()

        syn_idx = ['sample-'+str(i) for i in range(1, syn_x.shape[0]+1)]
        
        syn_feature = pd.DataFrame(syn_x, columns=train_feature.columns, index=syn_idx)
        syn_feature.index.name = 'Sample'
        syn_feature.to_csv(hparam_out_dir + '/synthetic_feature.tsv', sep='\t', index=True)
        
        syn_condition = pd.get_dummies(syn_y).astype(int)
        print('syn_condition', syn_condition.shape)
        syn_condition.columns = condition_cols
        syn_condition.index = syn_idx
        syn_condition.index.name = 'Sample'
        syn_condition.to_csv(hparam_out_dir + '/synthetic_condition.tsv', sep='\t', index=True)
        
        hparam_log_dir = log_dir + '/' + hparam_label[i]
        Path(hparam_log_dir).mkdir(exist_ok=True, parents=True)
        
        model = RandomForestClassifier(n_estimators=syn_x.shape[1])
        model = model.fit(syn_x, syn_y)
        pred_y = model.predict_proba(val_x)
        pred_df = pd.DataFrame(pred_y, columns=condition_cols, index=val_feature.index)
        pred_df.to_csv(hparam_log_dir + '/rf.predict_proba.tsv', sep='\t', index=True)
        auroc.append(roc_auc_score(val_y, pred_y[:, 1]))
    hparam_df['auroc'] = auroc
    hparam_df.to_csv(log_dir + '/hyperparameters.tsv', sep='\t', index=False)
    
    hparam_df = hparam_df.set_index('hparam_label')
    best_hparam = hparam_df['auroc'].idxmax()
    
    with open(log_dir + '/best_hparam.txt', 'w') as fp:
        fp.write(best_hparam)
        
        
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