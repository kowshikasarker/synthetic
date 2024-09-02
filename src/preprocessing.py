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
    parser.add_argument("--microbiome_path", type=str,
                        help="path to the micriobiome data in .tsv formnat",
                        required=True, default=None)
    parser.add_argument("--metabolome_path", type=str,
                        help="path to the metabolome data in .tsv formnat",
                        required=True, default=None)
    parser.add_argument("--metadata_path", type=str,
                        help="path to the metadata data in .tsv formnat",
                        required=True, default=None)
    parser.add_argument("--missing_pct", type=float,
                        help="percentage of missing values for filtering columns",
                        required=True, default=None)
    parser.add_argument("--imputation_method", type=str,
                        help="method to impute values",
                        required=True, default=None)
    parser.add_argument("--corr_method", type=str,
                        help="Correlation method to consider in the correlation network",
                        required=True, default=None)
    parser.add_argument("--corr_top_k", type=int,
                        help="top k edges to consider in the correlation network",
                        required=True, default=None)
    parser.add_argument("--mic_mic_prior_path", type=str,
                        help="Path to microbiome-microbiome prior edges in .tsv format",
                        required=True, default=None)
    parser.add_argument("--met_met_prior_path", type=str,
                        help="Path to metabolome-metabolome prior edges in .tsv format",
                        required=True, default=None)
    parser.add_argument("--mic_met_prior_path", type=str,
                        help="Path to microbiome-metabolome prior edges in .tsv format",
                        required=True, default=None)
    parser.add_argument("--prior_top_k", type=int,
                        help="top k edges to consider in the prior network",
                        required=True, default=None)
    parser.add_argument("--train_pct", type=float,
                        help="Percentage of sample for training data",
                        required=True, default=None)
    parser.add_argument("--out_dir", type=str,
                        help="path to the output dir",
                        required=True, default=None)
    args = parser.parse_args()
    return args

def knn_impute(df):
    imputer = KNNImputer(n_neighbors=5)
    df = pd.DataFrame(imputer.fit_transform(df.to_numpy()),
                      columns=df.columns,
                      index=df.index)
    return df

def impute(df, imputation_method):
    if(imputation_method == 'knn'):
        return knn_impute(df)
    

def preprocess_input(data_path, missing_pct, imputation_method, prefix):
    print('preprocess_input', data_path)
    
    df = pd.read_csv(data_path, sep='\t', index_col='Sample')
    df = df.add_prefix(prefix)
    df = df.replace(0, np.nan)
    
    mask = df.isnull().mean(axis=0)
    cols = list(mask[mask < missing_pct].index)
    df = df[cols]
    df = impute(df, imputation_method)
    
    df = df.div(df.sum(axis=1), axis=0) # row normalization
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df.to_numpy()),
                          columns=df.columns,
                          index=df.index)
    return df

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
    mic_mic_edges = get_top_prior_edges(mic_mic_prior_path, mic_set, mic_set, prior_top_k, 'mic:', 'mic:')
    met_met_edges = get_top_prior_edges(met_met_prior_path, met_set, met_set, prior_top_k, 'met:', 'met:')
    mic_met_edges = get_top_prior_edges(mic_met_prior_path, mic_set, met_set, prior_top_k, 'mic:', 'met:')
    
    prior_edges = pd.concat([mic_mic_edges, met_met_edges, mic_met_edges], axis=0)
    prior_edges = pd.DataFrame(np.sort(prior_edges.to_numpy(), axis=1),
                            columns=prior_edges.columns,
                            index=prior_edges.index)
    prior_edges = prior_edges.drop_duplicates()
    return prior_edges


def get_correlation_edges(mic_df, met_df, corr_method, corr_top_k):
    mic_cols = list(mic_df.columns)
    met_cols = list(met_df.columns)
    
    data_df = pd.concat([mic_df, met_df], axis=1)
    
    corr_df = data_df.corr(corr_method)
    corr_df.to_csv(corr_method + '_correlation.tsv', sep='\t', index=True)
    corr_df = corr_df.abs()
    np.fill_diagonal(corr_df.values, -100)
    
    mic_mic_corr = corr_df.loc[mic_cols, mic_cols]
    met_met_corr = corr_df.loc[met_cols, met_cols]
    mic_met_corr = corr_df.loc[mic_cols, met_cols]
    met_mic_corr = corr_df.loc[met_cols, mic_cols]
    
    def get_rowwise_top_columns(df, top_k):
        top_cols = ['Top-'+str(i) for i in range(1, top_k+1)]
        df = pd.DataFrame(df.apply(lambda x: x.nlargest(top_k).index.astype(str).tolist(), axis=1).tolist(), 
                               columns=top_cols, index=df.index)
        return df
    
    mic_mic_top_corr = get_rowwise_top_columns(mic_mic_corr, corr_top_k)
    met_met_top_corr = get_rowwise_top_columns(met_met_corr, corr_top_k)
    mic_met_top_corr = get_rowwise_top_columns(mic_met_corr, corr_top_k)
    met_mic_top_corr = get_rowwise_top_columns(met_mic_corr, corr_top_k)
    
    top_corr = pd.concat([mic_mic_top_corr,
                          met_met_top_corr,
                          mic_met_top_corr,
                          met_mic_top_corr],
                         axis=0)
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

def preprocess(microbiome_path, metabolome_path, metadata_path, missing_pct, imputation_method, corr_method, corr_top_k, mic_mic_prior_path, met_met_prior_path, mic_met_prior_path, prior_top_k, train_pct):
    
    mic_df = preprocess_input(microbiome_path, missing_pct, imputation_method, 'mic:')
    mic_df.to_csv('preprocessed_microbiome.tsv', sep='\t', index=True)
    mic_set = set(mic_df.columns.astype(str))
    
    met_df = preprocess_input(metabolome_path, missing_pct, imputation_method, 'met:')
    met_df.to_csv('preprocessed_metabolome.tsv', sep='\t', index=True)
    met_set = set(met_df.columns.astype(str))
    
    meta_df = pd.read_csv(metadata_path, sep='\t', index_col='Sample', usecols=['Sample', 'Study.Group'])
    meta_df = meta_df.rename(columns={'Study.Group': 'Cohort'})
    
    train_samples, test_samples, train_cohort, test_cohort = train_test_split(list(meta_df.index), list(meta_df.Cohort), train_size=train_pct, stratify=list(meta_df.Cohort))
    val_samples, test_samples, val_cohort, test_cohort = train_test_split(test_samples, test_cohort, test_size=0.50, stratify=test_cohort)
    
    print(len(train_samples), 'train_samples', len(val_samples), 'val_samples', len(test_samples), 'test_samples')
    
    train_mic_df = mic_df.loc[train_samples, :]
    train_mic_df.to_csv('train_microbiome.tsv', sep='\t', index=True)
    val_mic_df = mic_df.loc[val_samples, :]
    val_mic_df.to_csv('val_microbiome.tsv', sep='\t', index=True)
    test_mic_df = mic_df.loc[test_samples, :]
    test_mic_df.to_csv('test_microbiome.tsv', sep='\t', index=True)
    
    train_met_df = met_df.loc[train_samples, :]
    train_met_df.to_csv('train_metabolome.tsv', sep='\t', index=True)
    val_met_df = met_df.loc[val_samples, :]
    val_met_df.to_csv('val_metabolome.tsv', sep='\t', index=True)
    test_met_df = met_df.loc[test_samples, :]
    test_met_df.to_csv('test_metabolome.tsv', sep='\t', index=True)
    
    meta_df = pd.get_dummies(meta_df).astype(int)
    meta_df.to_csv('preprocessed_metadata.tsv', sep='\t', index=True)
    
    train_meta_df = meta_df.loc[train_samples, :]
    train_meta_df.to_csv('train_metadata.tsv', sep='\t', index=True)
    val_meta_df = meta_df.loc[val_samples, :]
    val_meta_df.to_csv('val_metadata.tsv', sep='\t', index=True)
    test_meta_df = meta_df.loc[test_samples, :]
    test_meta_df.to_csv('test_metadata.tsv', sep='\t', index=True)
    
    corr_edges = get_correlation_edges(mic_df, met_df, corr_method, corr_top_k)
    corr_edges.to_csv('correlation_edges.tsv', sep='\t', index=False)
    
    prior_edges = get_prior_edges(mic_mic_prior_path, met_met_prior_path, mic_met_prior_path, mic_set, met_set, prior_top_k)
    prior_edges.to_csv('prior_edges.tsv', sep='\t', index=False)
    
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
    
    preprocess(args.microbiome_path, args.metabolome_path, args.metadata_path, args.missing_pct, args.imputation_method, args.corr_method, args.corr_top_k, args.mic_mic_prior_path, args.met_met_prior_path, args.mic_met_prior_path, args.prior_top_k, args.train_pct)

    sys.stdout = original_stdout
    sys.stderr = original_stderr
    
    log_file.flush()
    log_file.close()

if __name__ == "__main__":
    main(parse_args())
