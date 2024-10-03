import argparse
from scipy.stats import kstest, spearmanr, f_oneway
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, average_precision_score, adjusted_rand_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import random
from sklearn.utils import shuffle
from pathlib import Path
import os, sys
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
    
    parser.add_argument("--test_mic_path", type=str,
                        help="path to the test micriobiome data in .tsv format",
                        required=True, default=None)
    parser.add_argument("--test_met_path", type=str,
                        help="path to the test metabolome data in .tsv format",
                        required=True, default=None)
    parser.add_argument("--test_meta_path", type=str,
                        help="path to the test metadata data in .tsv format",
                        required=True, default=None)
    
    parser.add_argument("--syn_mic_path", type=str,
                        help="path to the synthetic micriobiome data in .tsv format",
                        required=True, default=None)
    parser.add_argument("--syn_met_path", type=str,
                        help="path to the synthetic metabolome data in .tsv format",
                        required=True, default=None)
    parser.add_argument("--syn_meta_path", type=str,
                        help="path to the synthetic metadata data in .tsv format",
                        required=True, default=None)
    
    parser.add_argument("--out_dir", type=str,
                        help="path to output dir",
                        required=True, default=None)
    args = parser.parse_args()
    return args

def evaluate_mean_std_corr(exp_df, syn_df, out_dir):
    exp_mean = exp_df.mean(axis=0)
    exp_std = exp_df.std(axis=0)
    
    syn_mean = syn_df.mean(axis=0)
    syn_std = syn_df.std(axis=0)
    
    exp_mean, syn_mean = exp_mean.align(syn_mean, join='inner')
    exp_std, syn_std = exp_std.align(syn_std, join='inner')
    
    mean_corr = spearmanr(exp_mean, syn_mean)
    std_corr = spearmanr(exp_std, syn_std)
    
    df = pd.concat([exp_mean, syn_mean, exp_std, syn_std])
    print('df', df.shape)
    df.columns = ['exp_mean', 'syn_mean', 'exp_std', 'syn_std']
    df.to_csv(out_dir + '/mean_std.tsv', sep='\t', index=True)
    
    #mean_corr = exp_mean.corr(syn_mean, method='spearman')
    #std_corr = exp_std.corr(syn_std, method='spearman')
    
    mean_corr = spearmanr(exp_mean, syn_mean)
    std_corr = spearmanr(exp_std, syn_std)
    
    with open(out_dir + '/mean_std_corr.tsv', 'w') as fp:
        fp.write('aggr' + '\t' + 'statistic' + '\t' + 'p-value'+ '\n')
        fp.write('mean' + '\t' + str(mean_corr[0]) + '\t' + str(mean_corr[1]) + '\n')
        fp.write('std' + '\t' + str(std_corr[0]) + '\t' + str(std_corr[1]))
        

def evaluate_classifier(model, train_x, test_x, train_y, test_y):
    binary = True
    if ((len(set(train_y))) > 2):
        binary = False
    model = model.fit(train_x, train_y) 
    y_pred = model.predict_proba(test_x)
    if(binary):
        auroc = roc_auc_score(test_y, y_pred[:, 1])
        auprc = average_precision_score(test_y, y_pred[:, 1])
        return auroc, auprc
    else:
        auroc = roc_auc_score(test_y, y_pred, multi_class='ovr')
        auprc = average_precision_score(test_y, y_pred, average='weighted')
        return auroc, auprc

def evaluate_discriminative_score(exp_df, syn_df, out_dir):
    train_pct = 0.60
    exp_train_count = int(exp_df.shape[0] * train_pct)
    syn_train_count = int(syn_df.shape[0] * train_pct)
    
    print('exp_train_count', exp_train_count,
          'syn_train_count', syn_train_count)
    
    # shuffle
    exp_df = exp_df.sample(frac=1)
    syn_df = syn_df.sample(frac=1)
    
    exp_train = exp_df.iloc[:exp_train_count, :]
    exp_test = exp_df.iloc[exp_train_count:, :]
    
    syn_train = syn_df.iloc[:syn_train_count, :]
    syn_test = syn_df.iloc[syn_train_count:, :]
    
    train_df = pd.concat([exp_train, syn_train], axis=0)
    test_df = pd.concat([exp_test, syn_test], axis=0)
    
    train_x = train_df.to_numpy()
    test_x = test_df.to_numpy()
    
    train_y = np.array([1] * exp_train.shape[0] + [0] * syn_train.shape[0])
    test_y = np.array([1] * exp_test.shape[0] + [0] * syn_test.shape[0])
    
    train_x, train_y = shuffle(train_x, train_y)
    test_x, test_y = shuffle(test_x, test_y)
    
    print('exp_train', exp_train.shape,
          'syn_train', syn_train.shape,
          'train_df', train_df.shape,
          'train_y', train_y.shape)
    
    print('exp_test', exp_test.shape,
          'syn_test', syn_test.shape,
          'test_df', test_df.shape,
          'test_y', test_y.shape)
    
    records = []
    
    # random forest
    model = RandomForestClassifier(max_depth=None)
    auroc, auprc = evaluate_classifier(model, train_x, test_x, train_y, test_y)
    records.append({
        'model': 'rf',
        'auroc': auroc,
        'auprc': auprc
    })
    
    # multi layer perceptron
    model = MLPClassifier(hidden_layer_sizes=[train_x.shape[1]//2], tol=0, max_iter=2000)
    auroc, auprc = evaluate_classifier(model, train_x, test_x, train_y, test_y)
    records.append({
        'model': 'mlp',
        'auroc': auroc,
        'auprc': auprc
    })
    
    # gradient boosting
    model = GradientBoostingClassifier(max_depth=None)
    auroc, auprc = evaluate_classifier(model, train_x, test_x, train_y, test_y)
    records.append({
        'model': 'gb',
        'auroc': auroc,
        'auprc': auprc
    })
    
    # k nearest neighbors
    model = KNeighborsClassifier()
    auroc, auprc = evaluate_classifier(model, train_x, test_x, train_y, test_y)
    records.append({
        'model': 'knn',
        'auroc': auroc,
        'auprc': auprc
    })
    
    # naive bayes
    model = GaussianNB()
    auroc, auprc = evaluate_classifier(model, train_x, test_x, train_y, test_y)
    records.append({
        'model': 'nb',
        'auroc': auroc,
        'auprc': auprc
    })
    
    df = pd.DataFrame.from_records(records)
    df.to_csv(out_dir + '/discriminative_score.tsv', sep='\t')
    

def evaluate_kstest(exp_df, syn_df, out_dir):    
    res = kstest(exp_df.to_numpy(), syn_df.to_numpy(), axis=0)
    
    ks_df = pd.DataFrame(zip(exp_df.columns, res[0], res[1]),
                         columns=['column', 'statistic', 'p-value'])
    ks_df.to_csv(out_dir + '/kstest.tsv', sep='\t', index=False)
    
    with open(out_dir + '/kstest-pval.tsv', 'w') as fp:
        fp.write('mean' + '\t' + str(np.mean(res[1])) + '\n')
        fp.write('median' + '\t' + str(np.median(res[1])) + '\n')
        fp.write('minimum' + '\t' + str(np.min(res[1])) + '\n')
        fp.write('maximum' + '\t' + str(np.max(res[1])) + '\n')

def evaluate_fidelity(exp_df, syn_df, out_dir):
    evaluate_kstest(exp_df, syn_df, out_dir)
    evaluate_mean_std_corr(exp_df, syn_df, out_dir)
    evaluate_discriminative_score(exp_df, syn_df, out_dir)

def evaluate_correlation(exp_df, syn_df, out_dir):
    exp_corr = exp_df.corr(method='spearman')
    syn_corr = syn_df.corr(method='spearman')
    corr_diff = exp_corr.sub(syn_corr)
    
    exp_corr.to_csv(out_dir + '/exp_corr.tsv', sep='\t', index=True)
    syn_corr.to_csv(out_dir + '/syn_corr.tsv', sep='\t', index=True)
    corr_diff.to_csv(out_dir + '/corr_diff.tsv', sep='\t', index=True)
    
    corr_diff = corr_diff.abs()
    avg_diff = corr_diff.sum().sum() / (corr_diff.shape[0] * corr_diff.shape[1])
    
    
    
    def discretize_corr(corr_df):
        bins = [-1.0, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 1.0]
        labels = list(range(len(bins)-1))
        
        bin_df = pd.DataFrame()
        
        for col in corr_df.columns:
            bin_df[col] = pd.cut(corr_df[col],
                                 bins=bins,
                                 labels=labels,
                                 retbins=False,
                                 include_lowest=True)
        return bin_df
   
    exp_corr_bins = discretize_corr(exp_corr)
    syn_corr_bins = discretize_corr(syn_corr)
    
    corr_bins_match = exp_corr_bins.eq(syn_corr_bins)
    corr_accuracy = (corr_bins_match.sum().sum() / (corr_bins_match.shape[0] * corr_bins_match.shape[1])) * 100.0
    
    exp_corr_bins.to_csv(out_dir + '/exp_corr_bins.tsv', sep='\t', index=True)
    syn_corr_bins.to_csv(out_dir + '/syn_corr_bins.tsv', sep='\t', index=True)
    corr_bins_match.to_csv(out_dir + '/corr_bins_match.tsv', sep='\t', index=True)
    
    exp_corr = exp_corr.to_numpy().flatten()
    syn_corr = syn_corr.to_numpy().flatten()
    
    res = spearmanr(exp_corr, syn_corr)
    
    with open(out_dir + '/correlation.tsv', 'w') as fp:
        fp.write('avg_diff' + '\t' + str(avg_diff) + '\n')
        fp.write('corr_accuracy' + '\t' + str(corr_accuracy) + '\n')
        fp.write('corr_corr' + '\t' + str(res[0]) + '\n')
        fp.write('corr_corr_pval' + '\t' + str(res[1]) + '\n')
        
def evaluate_utility_for_classifier(model,
                                   exp_train_x, exp_train_y,
                                   syn_train_x, syn_train_y,
                                   exp_syn_train_x, exp_syn_train_y,
                                   syn_exp_train_x, syn_exp_train_y,
                                   exp_test_x, exp_test_y):
    
    trtr_auroc, trtr_auprc = evaluate_classifier(model, exp_train_x, exp_test_x, exp_train_y, exp_test_y)
    tstr_auroc, tstr_auprc = evaluate_classifier(model, syn_train_x, exp_test_x, syn_train_y, exp_test_y)
    tsrtr_auroc, tsrtr_auprc = evaluate_classifier(model, syn_exp_train_x, exp_test_x, syn_exp_train_y, exp_test_y)
    trstr_auroc, trstr_auprc = evaluate_classifier(model, exp_syn_train_x, exp_test_x, exp_syn_train_y, exp_test_y)

    record = {
        'model': 'rf',
        'trtr_auroc': trtr_auroc,
        'trtr_auprc': trtr_auprc,
        'tstr_auroc': tstr_auroc,
        'tstr_auprc': tstr_auprc,
        'tsrtr_auroc': tsrtr_auroc,
        'tsrtr_auprc': tsrtr_auprc,
        'trstr_auroc': trstr_auroc,
        'trstr_auprc': trstr_auprc
    }
    return record
        
def evaluate_classification_utility(exp_train_x, exp_train_y,
                                    syn_train_x, syn_train_y,
                                    exp_syn_train_x, exp_syn_train_y,
                                    syn_exp_train_x, syn_exp_train_y,
                                    exp_test_x, exp_test_y,
                                    out_dir):
    records = []
    models = {
        'rf': RandomForestClassifier(max_depth=None),
        'mlp': MLPClassifier(hidden_layer_sizes=[exp_train_x.shape[1]//2], tol=0, max_iter=2000),
        'gb': GradientBoostingClassifier(max_depth=None),
        'knn': KNeighborsClassifier(),
        'nb': GaussianNB()
    }
    for model_name, model in models.items():
        record = evaluate_utility_for_classifier(model,
                                                 exp_train_x, exp_train_y,
                                                 syn_train_x, syn_train_y,
                                                 exp_syn_train_x, exp_syn_train_y,
                                                 syn_exp_train_x, syn_exp_train_y,
                                                exp_test_x, exp_test_y)
        record['model'] = model_name
        records.append(record)
    df = pd.DataFrame.from_records(records)
    df.to_csv(out_dir + '/classification_utility.tsv', sep='\t')
    
def evaluate_kmeans_clustering(X, true_y):
    kmeans = KMeans(n_clusters=len(set(true_y)))
    pred_y = kmeans.fit_predict(X)
    r = adjusted_rand_score(true_y, pred_y)
    s = silhouette_score(X, pred_y)
    return r, s
    
def evaluate_clustering_utility(exp_x, exp_y, syn_x, syn_y, out_dir):
    assert len(set(exp_y)) == len(set(syn_y))
    exp_r, exp_s = evaluate_kmeans_clustering(exp_x, exp_y)
    syn_r, syn_s = evaluate_kmeans_clustering(syn_x, syn_y)
    
    with open(out_dir + '/kmeans_clustering.tsv', 'w') as fp:
        fp.write('metric' + '\t' + 'exp' + '\t' + 'syn' + '\n')
        fp.write('adj_rand' + '\t' + str(exp_r) + '\t' + str(syn_r) + '\n')
        fp.write('silhouette' + '\t' + str(exp_s) + '\t' + str(syn_s) + '\n')
    

def evaluate_anova_utility(exp_df, syn_df, exp_meta, syn_meta, out_dir):    
    cohorts = set(exp_meta)
    
    records = []
    
    exp_cohort_rows = {}
    syn_cohort_rows = {}
    
    for cohort in cohorts:
        exp_cohort_rows[cohort] = list(exp_meta[exp_meta == cohort].index)
        syn_cohort_rows[cohort] = list(syn_meta[syn_meta == cohort].index)
    
    
    for col in exp_df.columns:
        exp_cohorts = []
        syn_cohorts = []
        for cohort in cohorts:
            exp_cohorts.append(list(exp_df.loc[exp_cohort_rows[cohort], col]))
            syn_cohorts.append(list(syn_df.loc[syn_cohort_rows[cohort], col]))
            
        exp_res = f_oneway(*exp_cohorts)
        syn_res = f_oneway(*syn_cohorts)
        
        records.append({
            'column': col,
            'exp_pval': exp_res[1],
            'syn_pval': syn_res[1]
        })
        
    df = pd.DataFrame.from_records(records)
    df.to_csv(out_dir + '/anova_pvalue.tsv', sep='\t', index=False)
    
    
    exp_pval = df['exp_pval']
    syn_pval = df['syn_pval']
    
    exp_pval, syn_pval = exp_pval.align(syn_pval)
    
    res = spearmanr(exp_pval, syn_pval)
    with open(out_dir + '/anova_pvalue_corr.tsv', 'w') as fp:
        fp.write('corr' + '\t' + 'pvalue' + '\n')
        fp.write(str(res[0]) + '\t' + str(res[1]) + '\n')
    
def evaluate_utility(exp_df, syn_df, exp_meta, syn_meta, out_dir):
    col_map = dict(zip(exp_meta.columns, list(range(exp_meta.shape[1]))))
    exp_meta = exp_meta.rename(columns=col_map) 
    syn_meta = syn_meta.rename(columns=col_map) 
    exp_meta = exp_meta.idxmax(axis=1)
    syn_meta = syn_meta.idxmax(axis=1)
    meta_col = 'Cohort'
    exp_meta.name = meta_col
    syn_meta.name = meta_col
    print('exp_meta', exp_meta,
         'syn_meta', syn_meta)
    
    evaluate_anova_utility(exp_df, syn_df, exp_meta, syn_meta, out_dir)
    
    exp_df = pd.concat([exp_df, exp_meta], axis=1)
    syn_df = pd.concat([syn_df, syn_meta], axis=1)
    
    exp_df = exp_df.sample(frac=1)
    syn_df = syn_df.sample(frac=1)
    
    train_pct = 0.60
    train_count = int(exp_df.shape[0] * train_pct)
    
    exp_train, exp_test = train_test_split(exp_df, train_size=train_count, stratify=exp_df[meta_col])
    syn_train, _ = train_test_split(syn_df, train_size=train_count, stratify=syn_df[meta_col])
    exp_train_half, _ = train_test_split(exp_train, train_size=0.5, stratify=exp_train[meta_col])
    syn_train_half, _ = train_test_split(syn_train, train_size=0.5, stratify=syn_train[meta_col])
    exp_syn_train = pd.concat([exp_train, syn_train_half], axis=0)
    syn_exp_train = pd.concat([syn_train, exp_train_half], axis=0)
    
    exp_y = list(exp_df[meta_col])
    exp_x = exp_df.drop(columns=[meta_col]).to_numpy()
    syn_y = list(syn_df[meta_col])
    syn_x = syn_df.drop(columns=[meta_col]).to_numpy()
    
    exp_train_y = list(exp_train[meta_col])
    exp_train_x = exp_train.drop(columns=[meta_col]).to_numpy()
    
    exp_test_y = list(exp_test[meta_col])
    exp_test_x = exp_test.drop(columns=[meta_col]).to_numpy()
    
    syn_train_y = list(syn_train[meta_col])
    syn_train_x = syn_train.drop(columns=[meta_col]).to_numpy()
    
    exp_syn_train_y = list(exp_syn_train[meta_col])
    exp_syn_train_x = exp_syn_train.drop(columns=[meta_col]).to_numpy()
    
    syn_exp_train_y = list(syn_exp_train[meta_col])
    syn_exp_train_x = syn_exp_train.drop(columns=[meta_col]).to_numpy()
    
    evaluate_clustering_utility(exp_x, exp_y, syn_x, syn_y, out_dir)
    evaluate_classification_utility(exp_train_x, exp_train_y,
                                    syn_train_x, syn_train_y,
                                    exp_syn_train_x, exp_syn_train_y,
                                    syn_exp_train_x, syn_exp_train_y,
                                    exp_test_x, exp_test_y,
                                    out_dir)
    
def evaluate_membership_inference(train_df, test_df, syn_df, out_dir):
    train_df = train_df.sample(n=test_df.shape[0])
    train_test_df = pd.concat([train_df, test_df], axis=0)
    y_true = [1] * train_df.shape[0] + [0] * test_df.shape[0]
    
    model = NearestNeighbors(n_neighbors=1)
    model = model.fit(syn_df.to_numpy())
    y_score = model.kneighbors(train_test_df.to_numpy(), 1, return_distance=True)[0]
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    
    with open(out_dir + '/membership_inference.tsv', 'w') as fp:
        fp.write('auroc' + '\t' + str(auroc) + '\n')
        fp.write('auprc' + '\t' + str(auprc) + '\n')
    
def calc_re_identification_ratio(train_df1, train_df2, non_train_df1, non_train_df2):
    model1 = NearestNeighbors(n_neighbors=1)
    model1 = model1.fit(train_df1.to_numpy())
    ngh_idx1 = model1.kneighbors(non_train_df1.to_numpy(), 1, return_distance=False).flatten()
    
    model2 = NearestNeighbors(n_neighbors=1)
    model2 = model2.fit(train_df2.to_numpy())
    ngh_idx2 = model2.kneighbors(non_train_df2.to_numpy(), 1, return_distance=False).flatten()
    
    match_count = (ngh_idx1 == ngh_idx2).sum()
    ratio = match_count / train_df1.shape[0]
    
    return ratio
    
        
def evaluate_re_identification(train_df, test_df, syn_df, out_dir):
    columns = list(train_df.columns)
    random.shuffle(columns)
    split_idx = len(columns) // 2
    cols1 = columns[:split_idx]
    cols2 = columns[split_idx:]
    
    print('columns', len(columns),
         'cols1', len(cols1),
         'cols2', len(cols2))
    
    train_df1 = train_df[cols1]
    train_df2 = train_df[cols2]
    
    test_df1 = test_df[cols1]
    test_df2 = test_df[cols2]
    
    syn_df1 = syn_df[cols1]
    syn_df2 = syn_df[cols2]
    
    test_ratio = calc_re_identification_ratio(train_df1, train_df2, test_df1, test_df2)
    syn_ratio = calc_re_identification_ratio(train_df1, train_df2, syn_df1, syn_df2)
    
    with open(out_dir + '/re_identification.tsv', 'w') as fp:
        fp.write('data' + '\t' + 'ratio' + '\n')
        fp.write('test_data' + '\t' + str(test_ratio) + '\n')
        fp.write('syn_data' + '\t' + str(syn_ratio) + '\n')
        
    
def evaluate_privacy(train_df, test_df, syn_df, out_dir):
    evaluate_membership_inference(train_df, test_df, syn_df, out_dir)
    evaluate_re_identification(train_df, test_df, syn_df, out_dir)
    
def evaluate_modality(train_df, test_df, syn_df, train_meta, test_meta, syn_meta, out_dir):
    print('train_df', train_df.shape,
         'test_df', test_df.shape,
         'syn_df', syn_df.shape)
    
    cols = list(set(test_df.columns).intersection(set(syn_df.columns)))
    print('cols', len(cols))
    
    train_df = train_df[cols]
    test_df = test_df[cols]
    syn_df = syn_df[cols]
    
    
    fdir = out_dir + '/fidelity'
    Path(fdir).mkdir(parents=True, exist_ok=True)
    evaluate_fidelity(test_df, syn_df, fdir)
    
    
    cdir = out_dir + '/correlation'
    Path(cdir).mkdir(parents=True, exist_ok=True)
    evaluate_correlation(test_df, syn_df, cdir)
    
    
    udir = out_dir + '/utility'
    Path(udir).mkdir(parents=True, exist_ok=True)
    evaluate_utility(test_df, syn_df, test_meta, syn_meta, udir)
    
    
    pdir = out_dir + '/privacy'
    Path(pdir).mkdir(parents=True, exist_ok=True)
    evaluate_privacy(train_df, test_df, syn_df, pdir)
    

def evaluate(**kwargs):
    train_mic = pd.read_csv(kwargs['train_mic_path'], sep='\t', index_col='Sample')
    train_met = pd.read_csv(kwargs['train_met_path'], sep='\t', index_col='Sample')
    train_meta = pd.read_csv(kwargs['train_meta_path'], sep='\t', index_col='Sample')
    train_mic_met = pd.concat([train_mic, train_met], axis=1)
    
    test_mic = pd.read_csv(kwargs['test_mic_path'], sep='\t', index_col='Sample')
    test_met = pd.read_csv(kwargs['test_met_path'], sep='\t', index_col='Sample')
    test_meta = pd.read_csv(kwargs['test_meta_path'], sep='\t', index_col='Sample')
    test_mic_met = pd.concat([test_mic, test_met], axis=1)
    
    syn_mic = pd.read_csv(kwargs['syn_mic_path'], sep='\t', index_col='Sample')
    syn_met = pd.read_csv(kwargs['syn_met_path'], sep='\t', index_col='Sample')
    syn_meta = pd.read_csv(kwargs['syn_meta_path'], sep='\t', index_col='Sample')
    syn_mic_met = pd.concat([syn_mic, syn_met], axis=1)
    
    
    evaluate_modality(train_mic, test_mic, syn_mic, train_meta, test_meta, syn_meta, os.getcwd() + '/mic')
    evaluate_modality(train_met, test_met, syn_met, train_meta, test_meta, syn_meta, os.getcwd() + '/met')
    evaluate_modality(train_mic_met, test_mic_met, syn_mic_met, train_meta, test_meta, syn_meta, os.getcwd() + '/mic_met')

def main(args):
    print("evaluate.py")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(args.out_dir)
    
    log_file = open(args.out_dir + '/evaluate.log', 'w')
    
    original_stdout = sys.stdout
    sys.stdout = log_file
    
    original_stderr = sys.stderr
    sys.stderr = log_file
    
    print(args)
    
    kwargs = vars(args)
    del kwargs['out_dir']
    evaluate(**kwargs)

    sys.stdout = original_stdout
    sys.stderr = original_stderr
    
    log_file.flush()
    log_file.close()

if __name__ == "__main__":
    main(parse_args())
