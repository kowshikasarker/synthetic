# split dataset into train, val, test
# preprocess train: discard columns with too many missing values, impute missing values, row normalize and column standardize
# preprocess val and test: with the parameters obtained from train preprocessing

import argparse
import pandas as pd
import pytaxonkit
from pathlib import Path
import os, sys
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
import networkx as nx
from torch_geometric.data import HeteroData
from sklearn.model_selection import train_test_split

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
    parser.add_argument("--missing_pct", type=float,
                        help="max percentage of missing values to keep microbiome or metabolome columns",
                        required=True, default=None)
    parser.add_argument("--imputation_method", type=str,
                        help="method to impute values",
                        required=True, default=None)
    parser.add_argument("--corr_method", type=str,
                        choices=['spearman', 'pearson'],
                        help="correlation method to consider in the correlation network",
                        required=True, default=None)
    parser.add_argument("--corr_top_k", type=int,
                        help="top k edges to consider in the correlation network",
                        required=True, default=None)
    parser.add_argument("--mic_mic_prior_path", type=str,
                        help="path to microbiome-microbiome prior edges in .tsv format",
                        required=True, default=None)
    parser.add_argument("--met_met_prior_path", type=str,
                        help="path to metabolome-metabolome prior edges in .tsv format",
                        required=True, default=None)
    parser.add_argument("--mic_met_prior_path", type=str,
                        help="path to microbiome-metabolome prior edges in .tsv format",
                        required=True, default=None)
    parser.add_argument("--prior_top_k", type=int,
                        help="top k edges to consider in the prior network",
                        required=True, default=None)
    parser.add_argument("--train_pct", type=float,
                        help="percentage of sample for training data",
                        required=True, default=None)
    #parser.add_argument("--val_pct", type=float,
                        help="percentage of sample for validation data",
                        required=True, default=None)
    parser.add_argument("--out_dir", type=str,
                        help="path to the output dir",
                        required=True, default=None)
    args = parser.parse_args()
    return args

def get_top_prior_edges(prior_path, node1_set, node2_set, prior_top_k, node1_prefix, node2_prefix):
    df = pd.read_csv(prior_path, sep='\t')
    df.columns = ['Node1', 'Node2', 'Weight']
    df['Node1'] = node1_prefix + df['Node1'].astype(str)
    df['Node2'] = node2_prefix + df['Node2'].astype(str)
    df = df[(df['Node1'].isin(node1_set)) & (df['Node2'].isin(node2_set))]
    rev_df = df.copy()
    rev_df[['Node1', 'Node2']] = rev_df[['Node2', 'Node1']]
    df = pd.concat([df, rev_df], axis=0)
    df = df.sort_values(by='Weight', ascending=False)
    df = df.groupby('Node1').head(prior_top_k).reset_index(drop=True)
    df = df.drop(columns='Weight')
    df = pd.DataFrame(np.sort(df.to_numpy(), axis=1),
                            columns=df.columns,
                            index=df.index)
    df = df.drop_duplicates()
    return df
    
    
def get_prior_edges(mic_mic_prior_path, met_met_prior_path, mic_met_prior_path, mic_set, met_set, prior_top_k):
    print('get_prior_edges')
    mic_mic_edges = get_top_prior_edges(mic_mic_prior_path, mic_set, mic_set, prior_top_k, 'mic:', 'mic:')
    met_met_edges = get_top_prior_edges(met_met_prior_path, met_set, met_set, prior_top_k, 'met:', 'met:')
    mic_met_edges = get_top_prior_edges(mic_met_prior_path, mic_set, met_set, prior_top_k, 'mic:', 'met:')
    
    print('mic_mic_edges', mic_mic_edges.shape, 'met_met_edges', met_met_edges.shape, 'mic_met_edges', mic_met_edges.shape)
    print('mic_mic_edges', mic_mic_edges[mic_mic_edges.Node1 == mic_mic_edges.Node2])
    print('met_met_edges', met_met_edges[met_met_edges.Node1 == met_met_edges.Node2])
    print('mic_met_edges', mic_met_edges[mic_met_edges.Node1 == mic_met_edges.Node2])
    prior_edges = pd.concat([mic_mic_edges, met_met_edges, mic_met_edges], axis=0)
    prior_edges = pd.DataFrame(np.sort(prior_edges.to_numpy(), axis=1),
                            columns=prior_edges.columns,
                            index=prior_edges.index)
    prior_edges = prior_edges.drop_duplicates()
    return prior_edges


def get_correlation_edges(mic_df, met_df, corr_method, corr_top_k):
    print('mic_df isna', mic_df.isna().sum().sum())
    print('met_df isna', met_df.isna().sum().sum())
    
    mic_cols = list(mic_df.columns)
    met_cols = list(met_df.columns)
    
    data_df = pd.concat([mic_df, met_df], axis=1)
    print('mic_df', mic_df.shape, 'met_df', met_df.shape, 'data_df', data_df.shape)
    
    corr_df = data_df.corr(corr_method)
    corr_df.to_csv(corr_method + '_correlation.tsv', sep='\t', index=True)
    corr_df = corr_df.abs()
    np.fill_diagonal(corr_df.values, -10)
    corr_df.to_csv(corr_method + '_correlation_absolute.tsv', sep='\t', index=True)
    print()
    
    mic_mic_corr = corr_df.loc[mic_cols, mic_cols]
    met_met_corr = corr_df.loc[met_cols, met_cols]
    mic_met_corr = corr_df.loc[mic_cols, met_cols]
    met_mic_corr = corr_df.loc[met_cols, mic_cols]
    
    print('mic_mic_corr', mic_mic_corr.shape)
    print('met_met_corr', met_met_corr.shape)
    print('mic_met_corr', mic_met_corr.shape)
    print('met_mic_corr', met_mic_corr.shape)
    
    def get_rowwise_top_columns(df, top_k):
        top_cols = ['Top-'+str(i) for i in range(1, top_k+1)]
        print('df.shape', df.shape)
        df = pd.DataFrame(df.apply(lambda x: x.nlargest(top_k).index.astype(str).tolist(), axis=1).tolist(), 
                               columns=top_cols, index=df.index)
        return df
    
    mic_mic_top_corr = get_rowwise_top_columns(mic_mic_corr, corr_top_k)
    met_met_top_corr = get_rowwise_top_columns(met_met_corr, corr_top_k)
    mic_met_top_corr = get_rowwise_top_columns(mic_met_corr, corr_top_k)
    met_mic_top_corr = get_rowwise_top_columns(met_mic_corr, corr_top_k)
    
    print('mic_mic_top_corr', mic_mic_top_corr.shape)
    print('met_met_top_corr', met_met_top_corr.shape)
    print('mic_met_top_corr', mic_met_top_corr.shape)
    print('met_mic_top_corr', met_mic_top_corr.shape)
    
    top_corr = pd.concat([mic_mic_top_corr,
                          met_met_top_corr,
                          mic_met_top_corr,
                          met_mic_top_corr],
                         axis=0)
    print('top_corr', top_corr.shape)
    top_corr = top_corr.stack()
    top_corr = top_corr.droplevel(axis=0, level=1).reset_index()
    top_corr.columns = ['Node1', 'Node2']
    top_corr = pd.DataFrame(np.sort(top_corr.to_numpy(), axis=1),
                            columns=top_corr.columns,
                            index=top_corr.index)
    top_corr = top_corr.drop_duplicates()
    return top_corr
    
    
def degree_stat(G):
    degrees = [val for (node, val) in G.degree()]
    print(pd.Series(degrees).value_counts())
    print('min', min(degrees), 'max', max(degrees), 'median', np.median(degrees), 'mean', np.mean(degrees))

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

def impute(df, train_df, val_df, test_df, imputation_method):
    if(imputation_method == 'knn'):
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

def preprocess_input(df, train_df, val_df, test_df, train_meta_df, missing_pct, imputation_method):
    valid_cols = set()
    for cohort in train_meta_df.columns:
        cohort_samples = list((train_meta_df[train_meta_df[cohort] == 1]).index)
        print(len(cohort_samples), 'cohort_samples')
        cohort_train_df = train_df.loc[cohort_samples, :]
        print(cohort_train_df.shape, 'cohort_train_df')
        cohort_mask = cohort_train_df.isnull().mean(axis=0)
        cohort_cols = set(cohort_mask[cohort_mask < missing_pct].index)
        valid_cols.update(cohort_cols)
        print(len(valid_cols), 'valid_cols')
        
    valid_cols = list(valid_cols)
    print(len(valid_cols), 'valid_cols')
    
    df = df[valid_cols]
    train_df = train_df[valid_cols]
    val_df = val_df[valid_cols]
    test_df = test_df[valid_cols]
    
    df, train_df, val_df, test_df = impute(df, train_df, val_df, test_df, imputation_method)
    df, train_df, val_df, test_df = normalize(df, train_df, val_df, test_df)
    
    return df, train_df, val_df, test_df

#def preprocess(microbiome_path, metabolome_path, metadata_path, missing_pct, imputation_method, corr_method, corr_top_k, mic_mic_prior_path, met_met_prior_path, mic_met_prior_path, prior_top_k, train_pct):
    
def preprocess(**kwargs):    
    print('microbiome_path', kwargs['mic_path'])
    mic_df = pd.read_csv(kwargs['mic_path'], sep='\t', index_col='Sample')
    mic_df = mic_df.add_prefix('mic:')
    mic_df = mic_df.replace(0, np.nan)
    
    print('metabolome_path', kwargs['met_path'])
    met_df = pd.read_csv(kwargs['met_path'], sep='\t', index_col='Sample')
    met_df = met_df.add_prefix('met:')
    met_df = met_df.replace(0, np.nan)
    
    print('metadata_path', kwargs['meta_path'])
    meta_df = pd.read_csv(kwargs['meta_path'], sep='\t', index_col='Sample', usecols=['Sample', 'Study.Group'])
    meta_df = meta_df.rename(columns={'Study.Group': 'Cohort'})
    
    print('train_pct', kwargs['train_pct'])
    #train_count = int(kwargs['train_pct'] * meta_df.shape[0])
    #val_count = int(kwargs['val_pct'] * meta_df.shape[0])
    
    train_samples, test_samples, train_cohort, test_cohort = train_test_split(list(meta_df.index), list(meta_df.Cohort), train_size=kwargs['train_pct'], stratify=list(meta_df.Cohort))
    val_samples, test_samples, val_cohort, test_cohort = train_test_split(test_samples, test_cohort, test_size=0.50, stratify=test_cohort)
    
    print(len(train_samples), 'train_samples', len(val_samples), 'val_samples', len(test_samples), 'test_samples')
    
    print('train_cohort')
    print(pd.Series(train_cohort).value_counts())
    
    print('val_cohort')
    print(pd.Series(val_cohort).value_counts())
    
    print('test_cohort')
    print(pd.Series(test_cohort).value_counts())
    
    train_mic_df = mic_df.loc[train_samples, :]
    val_mic_df = mic_df.loc[val_samples, :]
    test_mic_df = mic_df.loc[test_samples, :]
    
    train_met_df = met_df.loc[train_samples, :]
    val_met_df = met_df.loc[val_samples, :]
    test_met_df = met_df.loc[test_samples, :]
    
    meta_df = pd.get_dummies(meta_df).astype(int)
    
    train_meta_df = meta_df.loc[train_samples, :]
    val_meta_df = meta_df.loc[val_samples, :]
    test_meta_df = meta_df.loc[test_samples, :]
    
    
    print('missing_pct', kwargs['missing_pct'], 'imputation_method', kwargs['imputation_method'])
    mic_df, train_mic_df, val_mic_df, test_mic_df = preprocess_input(mic_df, train_mic_df, val_mic_df, test_mic_df, train_meta_df, kwargs['missing_pct'], kwargs['imputation_method'])
    met_df, train_met_df, val_met_df, test_met_df = preprocess_input(met_df, train_met_df, val_met_df, test_met_df, train_meta_df, kwargs['missing_pct'], kwargs['imputation_method'])
    
    
    mic_df.to_csv('preprocessed_microbiome.tsv', sep='\t', index=True)
    train_mic_df.to_csv('train_microbiome.tsv', sep='\t', index=True)
    val_mic_df.to_csv('val_microbiome.tsv', sep='\t', index=True)
    test_mic_df.to_csv('test_microbiome.tsv', sep='\t', index=True)
    
    met_df.to_csv('preprocessed_metabolome.tsv', sep='\t', index=True)
    train_met_df.to_csv('train_metabolome.tsv', sep='\t', index=True)
    val_met_df.to_csv('val_metabolome.tsv', sep='\t', index=True)
    test_met_df.to_csv('test_metabolome.tsv', sep='\t', index=True)
    
    meta_df.to_csv('preprocessed_metadata.tsv', sep='\t', index=True)
    train_meta_df.to_csv('train_metadata.tsv', sep='\t', index=True)
    val_meta_df.to_csv('val_metadata.tsv', sep='\t', index=True)
    test_meta_df.to_csv('test_metadata.tsv', sep='\t', index=True)
    
    print('corr_method', kwargs['corr_method'], 'corr_top_k', kwargs['corr_top_k'])
    corr_edges = get_correlation_edges(train_mic_df, train_met_df, kwargs['corr_method'], kwargs['corr_top_k'])
    corr_edges.to_csv('correlation_edges.tsv', sep='\t', index=False)
    
    print('corr_edges duplicates')
    print(corr_edges[corr_edges.Node1 == corr_edges.Node2])
    
    mic_set = set(train_mic_df.columns.astype(str))
    met_set = set(train_met_df.columns.astype(str))
    
    print('mic_mic_prior_path', kwargs['mic_mic_prior_path'])
    print('met_met_prior_path', kwargs['met_met_prior_path'])
    print('mic_met_prior_path', kwargs['mic_met_prior_path'])
    prior_edges = get_prior_edges(kwargs['mic_mic_prior_path'], kwargs['met_met_prior_path'], kwargs['mic_met_prior_path'], mic_set, met_set, kwargs['prior_top_k'])
    prior_edges.to_csv('prior_edges.tsv', sep='\t', index=False)
    
    print('prior_edges duplicates')
    print(prior_edges[prior_edges.Node1 == prior_edges.Node2])
    
    edges = pd.concat([corr_edges, prior_edges], axis=0)
    edges = edges.drop_duplicates()
    edges.to_csv('edges.tsv', sep='\t', index=False)
    
    corr_ntw = nx.from_pandas_edgelist(corr_edges, source='Node1', target='Node2')
    prior_ntw = nx.from_pandas_edgelist(prior_edges, source='Node1', target='Node2')
    common_ntw = nx.intersection(corr_ntw, prior_ntw)
    
    print('Correlation network')
    print(corr_ntw.number_of_nodes(), 'Nodes', corr_ntw.number_of_edges(), 'Edges', nx.number_of_isolates(corr_ntw), 'isolates')
    corr_ntw.remove_nodes_from(list(nx.isolates(corr_ntw)))
    degree_stat(corr_ntw)
    
    print('Prior network')
    print(prior_ntw.number_of_nodes(), 'Nodes', prior_ntw.number_of_edges(), 'Edges', nx.number_of_isolates(prior_ntw), 'isolates')
    prior_ntw.remove_nodes_from(list(nx.isolates(prior_ntw)))
    degree_stat(prior_ntw)
    
    print('Common network')
    print(common_ntw.number_of_nodes(), 'Nodes', common_ntw.number_of_edges(), 'Edges', nx.number_of_isolates(common_ntw), 'isolates')
    common_ntw.remove_nodes_from(list(nx.isolates(common_ntw)))
    degree_stat(common_ntw)
    
def main(args):
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(args.out_dir)
    
    log_file = open(args.out_dir + '/preprocessing.log', 'w')
    
    original_stdout = sys.stdout
    sys.stdout = log_file
    
    original_stderr = sys.stderr
    sys.stderr = log_file
    
    print(args)
    
    #preprocess(args.microbiome_path, args.metabolome_path, args.metadata_path, args.missing_pct, args.imputation_method, args.corr_method, args.corr_top_k, args.mic_mic_prior_path, args.met_met_prior_path, args.mic_met_prior_path, args.prior_top_k, args.train_pct)
    
    kwargs = vars(args)
    del kwargs['out_dir']
    preprocess(**kwargs)

    sys.stdout = original_stdout
    sys.stderr = original_stderr
    
    log_file.flush()
    log_file.close()

if __name__ == "__main__":
    main(parse_args())
