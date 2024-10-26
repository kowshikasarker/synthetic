import os, sys, argparse, dcor

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from pathlib import Path
from math import ceil
from shutil import rmtree

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, pearsonr
from scipy.stats import f_oneway

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path", type=str,
                        help="path to the features in .tsv format, the first column name should be 'Sample' containing sample identifiers and the rest of the columns should contain metabolomic concentrations",
                        required=True, default=None)
    parser.add_argument("--condition_path", type=str,
                        help="path to the one-hot encoded disease status in .tsv format, the first column name should be 'Sample' containing sample identifiers, and the rest of the columns should each denote one disease group and contain either 0 or 1",
                        required=True, default=None)
    
    parser.add_argument("--missing_pct", type=float,
                        help="max percentage of missing values to keep feature columns, only columns which have less missing percentage in at least one disease group are kept",
                        required=True, default=None)
    parser.add_argument("--imputation", type=str,
                        choices=['knn'],
                        help="method to impute values, currently only knn imputation is supported",
                        required=True, default=None)
    
    parser.add_argument("--feature_count", type=int, 
                        help="no. of metabolites to keep based on the least anova p-values across disease groups",
                        required=True, default=None)
    parser.add_argument("--train_pct", type=float,
                        help="percentage of sample for training data, used to train the generative model",
                        required=True, default=None)
    parser.add_argument("--val_pct", type=float,
                        help="percentage of sample for validation data, used to tune hyperparameters of the generative model",
                        required=True, default=None)
    
    parser.add_argument("--corr_method", type=str, nargs='+',
                        choices=['sp', 'pr', 'dcov', 'dcol'],
                        help="corr measure(s) to use in constructing correlation graph for samples, needs at least one method",
                        required=False, default=None)
    parser.add_argument("--corr_top_pct", type=float,
                        help="top percentage of correlation edges to use, for every metabolite this percentage of top correlated metabolites are conencted with correlation edges",
                        required=False, default=None)
    
    parser.add_argument("--prior_path", type=str,
                        help="path to prior edges in .tsv format, should contain three columns, the first two contianing metabolites and the third column containing the weight of the prior connection",
                        required=False, default=None)
    parser.add_argument("--prior_top_pct", type=float,
                        help="top percentage of prior edges to use, for every metabolite this percentage of top prior edges are kept based on higehr weights",
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
    
    def pr_corr(a, b):
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
    
def row_normalize(df, train_df, val_df, test_df):
    df = df.div(df.sum(axis=1), axis=0) # row normalization
    train_df = train_df.div(train_df.sum(axis=1), axis=0) # row normalization
    val_df = val_df.div(val_df.sum(axis=1), axis=0)
    test_df = test_df.div(test_df.sum(axis=1), axis=0)
    
    return df, train_df, val_df, test_df

def col_standardize(df, train_df, val_df, test_df):    
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

def preprocess_input(df,
                     train_df,
                     val_df,
                     test_df,
                     train_meta_df,
                     feature_count,
                     missing_pct,
                     imputation):
    valid_cols = set()
    
    print('df', df.shape,
          'train_df', train_df.shape,
          'val_df', val_df.shape,
          'test_df', test_df.shape,
          'train_meta_df', train_meta_df.shape)
    
    for cohort in train_meta_df.columns:
        cohort_samples = list(train_meta_df[train_meta_df[cohort] == 1].index)
        cohort_train_df = train_df.loc[cohort_samples, :]
        cohort_mask = cohort_train_df.isnull().mean(axis=0)
        cohort_cols = set(cohort_mask[cohort_mask < missing_pct].index)
        valid_cols.update(cohort_cols)
        
    valid_cols = list(valid_cols)
    valid_cols.sort()
    print(len(valid_cols), 'valid_cols')     
    
    df = df[valid_cols]
    train_df = train_df[valid_cols]
    val_df = val_df[valid_cols]
    test_df = test_df[valid_cols]
    
    df = df.fillna(0)
    train_df = train_df.fillna(0)
    val_df = val_df.fillna(0)
    test_df = test_df.fillna(0)
    
    df, train_df, val_df, test_df = row_normalize(df, train_df, val_df, test_df)
    print('After first normalization', end='\n')
    print('df', '\n', df.sum(axis=1))
    print('train_df', '\n', train_df.sum(axis=1))
    print('val_df', '\n', val_df.sum(axis=1))
    print('test_df', '\n', test_df.sum(axis=1))
    
    df = df.replace(0, np.nan)
    train_df = train_df.replace(0, np.nan)
    val_df = val_df.replace(0, np.nan)
    test_df = test_df.replace(0, np.nan)
    
    df, train_df, val_df, test_df = impute(df, train_df, val_df, test_df, imputation)
    print('After imputation', end='\n')
    print('df', '\n', df.sum(axis=1))
    print('train_df', '\n', train_df.sum(axis=1))
    print('val_df', '\n', val_df.sum(axis=1))
    print('test_df', '\n', test_df.sum(axis=1))
    
    df, train_df, val_df, test_df = row_normalize(df, train_df, val_df, test_df)
    print('After second normalization', end='\n')
    print('df', '\n', df.sum(axis=1))
    print('train_df', '\n', train_df.sum(axis=1))
    print('val_df', '\n', val_df.sum(axis=1))
    print('test_df', '\n', test_df.sum(axis=1))
    
    df, train_df, val_df, test_df = col_standardize(df, train_df, val_df, test_df)
    
    cohorts = []
    for cohort in train_meta_df.columns:
        cohort_samples = list(train_meta_df[train_meta_df[cohort] == 1].index)
        cohort_train_df = train_df.loc[cohort_samples, valid_cols]
        cohorts.append(cohort_train_df.to_numpy())
    stat, pval = f_oneway(*cohorts, axis=0)
    print('pval', len(pval))
    print(pval)
    anova = pd.DataFrame(zip(valid_cols, stat, pval), columns=['Feature', 'Statistic', 'P-value'])
    anova = anova.sort_values(by='P-value', ascending=True)
    anova.to_csv('anova.tsv', sep='\t', index=False)
    final_cols = list(anova['Feature'])[:feature_count]
    print(len(final_cols), 'final_cols')        
    
    df = df[final_cols]
    train_df = train_df[final_cols]
    val_df = val_df[final_cols]
    test_df = test_df[final_cols]
    
    print('df', df.shape,
          'train_df', train_df.shape,
          'val_df', val_df.shape,
          'test_df', test_df.shape,
          'train_meta_df', train_meta_df.shape)
    
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
        
    train_count = int(round(kwargs['train_pct'] * condition_df.shape[0]))
    val_count = int(round(kwargs['val_pct'] * condition_df.shape[0]))
    
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
                                                                                     kwargs['feature_count'],
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
    
    edges = []
    
    for corr_method in kwargs['corr_method']:
        corr_edges = get_corr_edges(feature_df, corr_method, kwargs['corr_top_pct'])
        edges.append(corr_edges)
        corr_edges['edge_type'] = corr_method
        corr_edges.to_csv(corr_method + '_edges.tsv', sep='\t', index=False)
        print(corr_method + '_edges', corr_edges.shape)

    if (kwargs['prior_top_pct'] > 0):
        prior_edges = get_prior_edges(kwargs['prior_path'], set(feature_df.columns), kwargs['prior_top_pct'])
        edges.append(prior_edges)
        prior_edges['edge_type'] = 'prior'
        prior_edges.to_csv('prior_edges.tsv', sep='\t', index=False)
        print('prior_edges', prior_edges.shape)

        print('prior_edges duplicates')
        print(prior_edges[prior_edges.Node1 == prior_edges.Node2])
    
    edges = pd.concat(edges, axis=0)
    edges = edges.drop_duplicates()
    edges.to_csv('edges.tsv', sep='\t', index=False)
    
    G = nx.from_pandas_edgelist(edges, 'Node1', 'Node2')
    print(G.number_of_nodes(), 'nodes', G.number_of_edges(), 'edges')
    G = G.to_undirected()
    print(G.number_of_nodes(), 'nodes', G.number_of_edges(), 'edges')
    
    print('degree', dict(G.degree()))
    degrees = [val for (node, val) in G.degree()]
    plt.hist(degrees)
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree distribution of the undirected sample graph')
    plt.savefig('degree.png') 
    plt.close()
    
def main(args):
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
