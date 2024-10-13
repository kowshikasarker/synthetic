# added subparsers

import argparse
import pandas as pd
import pytaxonkit
from pathlib import Path
import os, sys
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from math import ceil
from scipy.stats import spearmanr, pearsonr
import dcor
from shutil import rmtree

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path", type=str,
                        help="path to the features in .tsv format",
                        required=True, default=None)
    parser.add_argument("--condition_path", type=str,
                        help="path to the condiiton in .tsv format",
                        required=True, default=None)
    
    parser.add_argument("--missing_pct", type=float,
                        help="max percentage of missing values to keep feature columns",
                        required=True, default=None)
    parser.add_argument("--imputation", type=str,
                        choices=['knn'],
                        help="method to impute values",
                        required=True, default=None)
    
    parser.add_argument("--train_pct", type=float,
                        help="percentage of sample for training data",
                        required=True, default=None)
    parser.add_argument("--val_pct", type=float,
                        help="percentage of sample for validation data",
                        required=True, default=None)
    
    parser.add_argument("--corr_method", type=str, choices=['sp', 'pr'],
                        help="correlation method to contruct the correlation network",
                        required=False, default='spearman')
    parser.add_argument("--corr_top_pct", type=float,
                        help="top k edges to consider in the correlation network",
                        required=False, default=None)
    
    parser.add_argument("--prior_path", type=str,
                        help="path to prior edges in .tsv format",
                        required=False, default=None)
    parser.add_argument("--prior_top_pct", type=float,
                        help="top k edges to consider in the prior network",
                        required=False, default=None)
    
    parser.add_argument("--out_dir", type=str,
                        help="path to the output dir",
                        required=True, default=None)
    
    args = parser.parse_args()
    return args
    
    
def get_prior_edges(prior_path, node_set, prior_top_pct):
    print('get_prior_edges')
    df = pd.read_csv(prior_path, sep='\t')
    df.columns = ['Node1', 'Node2', 'Weight']
    df = df[(df['Node1'].isin(node_set)) & (df['Node2'].isin(node_set))]

    prior_top_cnt = ceil(len(set(df['Node1']).union(set(df['Node2']))) * prior_top_pct)
    print('prior_top_cnt', prior_top_cnt)

    rev_df = df.copy()
    rev_df[['Node1', 'Node2']] = rev_df[['Node2', 'Node1']]
    df = pd.concat([df, rev_df], axis=0)
    df = df.drop_duplicates(subset=['Node1', 'Node2'])

    df = df.sort_values(by='Weight', ascending=False)
    df = df.groupby('Node1').head(prior_top_cnt).reset_index(drop=True)
    df = df.drop(columns='Weight')
    df = pd.DataFrame(np.sort(df.to_numpy(), axis=1),
                            columns=df.columns,
                            index=df.index)
    df = df.drop_duplicates()
    return df

def get_corr_edges(feature_df, corr_method, corr_top_pct):
    # smaller value means greater dependence/correlation
    # takes smallest connections
    
    assert corr_method in ('sp', 'pr', 'dcol', 'dcov')
    
    print('get_corr_edges')
    print('corr_method', corr_method, flush=True)
    
    def dcov_corr(a, b):
        print('dcov_corr', flush=True)
        return dcor.distance_covariance(a, b)
    
    def dcol_corr(a, b):
        print('dcol_corr', flush=True)
        df = pd.DataFrame(zip(a, b), columns=['a', 'b'])
        
        df = df.sort_values(by='a')
        b1 = df.iloc[:-1, :]['b'].to_numpy()
        b2 = df.iloc[1:, :]['b'].to_numpy()
        print('b1', b1.shape, 'b2', b2.shape)
        b_dist = np.subtract(b2, b1)
        b_dist = np.abs(b_dist)
        b_dist = np.mean(b_dist)
        return b_dist
    
    def sp_corr(a, b):
        print('sp_corr', flush=True)
        return -abs(spearmanr(a, b)[0])
    
    def pr_corr():
        print('pr_corr', flush=True)
        return -abs(pearsonr(a, b)[0])
    
    corr_func = {
        'sp': sp_corr,
        'pr': pr_corr,
        'dcov': dcov_corr,
        'dcol': dcol_corr
    }
    
    
    pairwise_corr = []
    
    cols = list(feature_df.columns)    
    func = corr_func[corr_method]
    for i in range(len(cols)):
        row = []
        for j in range(len(cols)):
            print('i', i, 'j', j, flush=True)
            if(i == j):
                row.append(np.inf)
            else:
                row.append(func(feature_df[cols[i]].to_numpy(), feature_df[cols[j]].to_numpy()))
        pairwise_corr.append(row)
    corr_df = pd.DataFrame(pairwise_corr, index=cols, columns=cols)
    corr_df.to_csv(corr_method + '_correlation.tsv', sep='\t', index=True)
    
    corr_top_cnt = ceil(corr_top_pct * feature_df.shape[1])
    print('corr_top_cnt', corr_top_cnt, flush=True)
    
    top_cols = ['Top-' + str(i) for i in range(1, corr_top_cnt+1)]
    top_corr = pd.DataFrame(corr_df.apply(lambda x: x.nsmallest(corr_top_cnt).index.astype(str).tolist(), axis=1).tolist(), 
                               columns=top_cols, index=corr_df.index)
    top_corr = top_corr.stack()
    top_corr = top_corr.droplevel(axis=0, level=1).reset_index()
    top_corr.columns = ['Node1', 'Node2']
    top_corr = pd.DataFrame(np.sort(top_corr.to_numpy(), axis=1),
                            columns=top_corr.columns,
                            index=top_corr.index)
    # the sorting is done so that duplicate edges like (Node1, Node2) and (Node2, Node1) become identical and later gets removed by drop_duplicates
    top_corr = top_corr.drop_duplicates()
    loop = top_corr[top_corr['Node1'] == top_corr['Node2']]
    assert loop.empty
    return top_corr

def knn_impute(df, train_df, val_df, test_df):
    imputer = KNNImputer()
    imputer = imputer.fit(train_df.to_numpy())
    
    df = pd.DataFrame(imputer.transform(df.to_numpy()),
                      columns=df.columns,
                      index=df.index)
    
    train_df = pd.DataFrame(imputer.transform(train_df.to_numpy()),
                      columns=train_df.columns,
                      index=train_df.index)
    
    val_df = pd.DataFrame(imputer.transform(val_df.to_numpy()),
                      columns=val_df.columns,
                      index=val_df.index)
    
    test_df = pd.DataFrame(imputer.transform(test_df.to_numpy()),
                      columns=test_df.columns,
                      index=test_df.index)
    
    return df, train_df, val_df, test_df

def impute(df, train_df, val_df, test_df, imputation):
    if(imputation == 'knn'):
        return knn_impute(df, train_df, val_df, test_df)    
    
def normalize(df, train_df, val_df, test_df):
    df = df.div(df.sum(axis=1), axis=0) # row normalization
    train_df = train_df.div(train_df.sum(axis=1), axis=0) # row normalization
    val_df = val_df.div(val_df.sum(axis=1), axis=0)
    test_df = test_df.div(test_df.sum(axis=1), axis=0)
    
    scaler = StandardScaler()
    scaler = scaler.fit(train_df.to_numpy())
    
    df = pd.DataFrame(scaler.transform(df.to_numpy()),
                      columns=df.columns,
                      index=df.index)
    
    train_df = pd.DataFrame(scaler.transform(train_df.to_numpy()),
                            columns=train_df.columns,
                            index=train_df.index)
    
    val_df = pd.DataFrame(scaler.transform(val_df.to_numpy()),
                          columns=val_df.columns,
                          index=val_df.index)
    
    test_df = pd.DataFrame(scaler.transform(test_df.to_numpy()),
                          columns=test_df.columns,
                          index=test_df.index)
    
    return df, train_df, val_df, test_df

def preprocess_input(df, train_df, val_df, test_df, train_meta_df, missing_pct, imputation):
    valid_cols = set()
    for cohort in train_meta_df.columns:
        cohort_samples = list(train_meta_df[train_meta_df[cohort] == 1].index)
        print(len(cohort_samples), 'cohort_samples')
        cohort_train_df = train_df.loc[cohort_samples, :]
        print(cohort_train_df.shape, 'cohort_train_df')
        cohort_mask = cohort_train_df.isnull().mean(axis=0)
        cohort_cols = set(cohort_mask[cohort_mask < missing_pct].index)
        valid_cols.update(cohort_cols)
        print(len(valid_cols), 'valid_cols')
        
    valid_cols = list(valid_cols)
    valid_cols.sort()
    print('total', len(valid_cols), 'valid_cols')
    
    df = df[valid_cols]
    train_df = train_df[valid_cols]
    val_df = val_df[valid_cols]
    test_df = test_df[valid_cols]
    
    df, train_df, val_df, test_df = impute(df, train_df, val_df, test_df, imputation)
    df, train_df, val_df, test_df = normalize(df, train_df, val_df, test_df)
    
    return df, train_df, val_df, test_df
    
def preprocess(**kwargs):
    feature_df = pd.read_csv(kwargs['feature_path'], sep='\t', index_col='Sample')
    feature_df = feature_df.add_prefix('feat:', axis=1)
    feature_df.index = feature_df.index.map(str)
    feature_df = feature_df.add_prefix('sample:', axis=0)
    feature_df = feature_df.replace(0, np.nan)
    
    condition_df = pd.read_csv(kwargs['condition_path'], sep='\t', index_col='Sample', usecols=['Sample', 'Study.Group'])
    condition_df.index = condition_df.index.map(str)
    condition_df = condition_df.add_prefix('sample:', axis=0)
    condition_df = condition_df.rename(columns={'Study.Group': 'Cohort'})
    condition_df['Cohort'] = condition_df['Cohort'].astype(str)
        
    train_count = int(kwargs['train_pct'] * condition_df.shape[0])
    val_count = int(kwargs['val_pct'] * condition_df.shape[0])
    
    train_samples, test_samples, train_cohort, test_cohort = train_test_split(list(condition_df.index),
                                                                              list(condition_df.Cohort),
                                                                              train_size=train_count,
                                                                              stratify=list(condition_df.Cohort),
                                                                              random_state=0)
    val_samples, test_samples, val_cohort, test_cohort = train_test_split(test_samples,
                                                                          test_cohort,
                                                                          train_size=val_count,
                                                                          stratify=test_cohort,
                                                                          random_state=0)
    
    print(len(train_samples), 'train_samples', len(val_samples), 'val_samples', len(test_samples), 'test_samples')
    
    print('train_cohort')
    print(pd.Series(train_cohort).value_counts())
    
    print('val_cohort')
    print(pd.Series(val_cohort).value_counts())
    
    print('test_cohort')
    print(pd.Series(test_cohort).value_counts())
    
    train_feature_df = feature_df.loc[train_samples, :]
    val_feature_df = feature_df.loc[val_samples, :]
    test_feature_df = feature_df.loc[test_samples, :]
    
    print('train_feature_df', train_feature_df.shape,
          'val_feature_df', val_feature_df.shape,
          'test_feature_df', test_feature_df.shape)
    
    condition_df = pd.get_dummies(condition_df).astype(int)
    
    print('condition_df', condition_df.head())
    
    train_condition_df = condition_df.loc[train_samples, :]
    val_condition_df = condition_df.loc[val_samples, :]
    test_condition_df = condition_df.loc[test_samples, :]
    
    print('train_condition_df', train_condition_df.shape,
          'val_condition_df', val_condition_df.shape,
          'test_condition_df', test_condition_df.shape)
    
    feature_df, train_feature_df, val_feature_df, test_feature_df = preprocess_input(feature_df,
                                                                     train_feature_df,
                                                                     val_feature_df,
                                                                     test_feature_df,
                                                                     train_condition_df,
                                                                     kwargs['missing_pct'],
                                                                     kwargs['imputation'])
    
    print('train_feature_df', train_feature_df.shape,
          'val_feature_df', val_feature_df.shape,
          'test_feature_df', test_feature_df.shape)
    
    feature_df.to_csv('preprocessed_feature.tsv', sep='\t', index=True)
    train_feature_df.to_csv('train_feature.tsv', sep='\t', index=True)
    val_feature_df.to_csv('val_feature.tsv', sep='\t', index=True)
    test_feature_df.to_csv('test_feature.tsv', sep='\t', index=True)
    
    condition_df.to_csv('preprocessed_condition.tsv', sep='\t', index=True)
    train_condition_df.to_csv('train_condition.tsv', sep='\t', index=True)
    val_condition_df.to_csv('val_condition.tsv', sep='\t', index=True)
    test_condition_df.to_csv('test_condition.tsv', sep='\t', index=True)
        
    corr_edges = get_corr_edges(feature_df, kwargs['corr_method'], kwargs['corr_top_pct'])
    corr_edges.to_csv('corr_edges.tsv', sep='\t', index=False)
    
    dcov_edges = get_corr_edges(feature_df, 'dcov', kwargs['corr_top_pct'])
    dcov_edges.to_csv('dcov_edges.tsv', sep='\t', index=False)
    
    dcol_edges = get_corr_edges(feature_df, 'dcol', kwargs['corr_top_pct'])
    dcol_edges.to_csv('dcol_edges.tsv', sep='\t', index=False)    

    prior_edges = get_prior_edges(kwargs['prior_path'], set(feature_df.columns), kwargs['prior_top_pct'])
    prior_edges.to_csv('prior_edges.tsv', sep='\t', index=False)

    print('prior_edges duplicates')
    print(prior_edges[prior_edges.Node1 == prior_edges.Node2])
    
    edges = pd.concat([corr_edges, dcov_edges, dcol_edges, prior_edges], axis=0)
    edges = edges.drop_duplicates()
    edges.to_csv('edges.tsv', sep='\t', index=False)
    
def main(args):
    if(os.path.exists(args.out_dir)):
        rmtree(args.out_dir)
    
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(args.out_dir)
    
    log_file = open(args.out_dir + '/preprocess.log', 'w')
    
    original_stdout = sys.stdout
    sys.stdout = log_file
    
    original_stderr = sys.stderr
    sys.stderr = log_file
    
    print('========== preprocess.py ==========')
    
    print(args)
    
    kwargs = vars(args)
    del kwargs['out_dir']
    preprocess(**kwargs)

    sys.stdout = original_stdout
    sys.stderr = original_stderr
    
    log_file.flush()
    log_file.close()

if __name__ == "__main__":
    main(parse_args())
