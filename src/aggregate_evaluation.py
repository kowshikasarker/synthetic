from pathlib import Path
import pandas as pd

summary_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Result/Summary'
base_result_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Result'

methods = ['PRADA', 'CVAE', 'NOISE', 'SMOTE']
    
method_dirs = {
    'PRADA': ['config-' + str(i) for i in range(1, 17)],
    'CVAE': ['separate_hidden', 'combined_hidden'],
    'NOISE': ['normal', 'uniform'],
    'SMOTE': ['']
}

method_hparam = {
    'PRADA': True,
    'CVAE': True,
    'NOISE': False,
    'SMOTE': False
}

datasets = [
    'YACHIDA_CRC_2019',
    'iHMP_IBDMDB_2019',
    'FRANZOSA_IBD_2019',
    'ERAWIJANTARI_GASTRIC_CANCER_2020',
    'MARS_IBS_2020',
    'KOSTIC_INFANTS_DIABETES_2015',
    'JACOBS_IBD_FAMILIES_2016',
    'KIM_ADENOMAS_2020'
]

'''datasets = [
    'SINHA_CRC_2016']'''

omics = [
    'mic',
    'met',
    'mic_met'
]

# substitution
sub_dir = summary_dir + '/substitution'
Path(sub_dir).mkdir(exist_ok=True, parents=True)

# kstest
      
# fidelity
fdir = sub_dir + '/fidelity'
Path(fdir).mkdir(exist_ok=True, parents=True)

# kstest
for omic in omics:
    omic_dir = fdir + '/' + omic
    Path(omic_dir).mkdir(exist_ok=True, parents=True)
    keys = []
    dfs = []
    for dataset in datasets:
        print('dataset', dataset)
        for method in methods:
            print('method', method)
            method_result_dir = base_result_dir + '/' + method + '/' + dataset
            for subdir in method_dirs[method]:
                working_dir = method_result_dir + '/' + subdir + '/output'
                keys.append((dataset, method, subdir))
                if (method_hparam[method]):
                    best_hparam_path = open(working_dir + '/logs/csv_logs/best_hparam.txt', 'r')
                    best_hparam = best_hparam_path.readline().strip()
                    working_dir = working_dir + '/' + best_hparam
                working_dir = working_dir + '/evaluate/substitution/' + omic + '/fidelity'
                df = pd.read_csv(working_dir + '/kstest.tsv', sep='\t', index_col='column')
                dfs.append(df)
    df = pd.concat(dfs, axis=1, keys=keys)
    df.to_csv(omic_dir + '/kstest.tsv', sep='\t', index=True)
        
        
# kstest-pval
for omic in omics:
    omic_dir = fdir + '/' + omic
    Path(omic_dir).mkdir(exist_ok=True, parents=True)
    keys = []
    dfs = []
    for dataset in datasets:
        print('dataset', dataset)
        for method in methods:
            print('method', method)
            method_result_dir = base_result_dir + '/' + method + '/' + dataset
            for subdir in method_dirs[method]:
                working_dir = method_result_dir + '/' + subdir + '/output'
                keys.append((dataset, method, subdir))
                if (method_hparam[method]):
                    best_hparam_path = open(working_dir + '/logs/csv_logs/best_hparam.txt', 'r')
                    best_hparam = best_hparam_path.readline().strip()
                    working_dir = working_dir + '/' + best_hparam
                working_dir = working_dir + '/evaluate/substitution/' + omic + '/fidelity'
                df = pd.read_csv(working_dir + '/kstest-pval.tsv', sep='\t', index_col=0, header=None)
                
                dfs.append(df)
    df = pd.concat(dfs, axis=1, keys=keys)
    df = df.droplevel(level=3, axis=1)
    df.index.name = None
    df.to_csv(omic_dir + '/kstest-pval.tsv', sep='\t', index=True)
        
# mean_std_corr
for omic in omics:
    omic_dir = fdir + '/' + omic
    Path(omic_dir).mkdir(exist_ok=True, parents=True)
    keys = []
    dfs = []
    for dataset in datasets:
        print('dataset', dataset)
        for method in methods:
            print('method', method)
            method_result_dir = base_result_dir + '/' + method + '/' + dataset
            for subdir in method_dirs[method]:
                working_dir = method_result_dir + '/' + subdir + '/output'
                keys.append((dataset, method, subdir))
                if (method_hparam[method]):
                    best_hparam_path = open(working_dir + '/logs/csv_logs/best_hparam.txt', 'r')
                    best_hparam = best_hparam_path.readline().strip()
                    working_dir = working_dir + '/' + best_hparam
                working_dir = working_dir + '/evaluate/substitution/' + omic + '/fidelity'
                df = pd.read_csv(working_dir + '/mean_std_corr.tsv', sep='\t', index_col=0)
                dfs.append(df)
    df = pd.concat(dfs, axis=1, keys=keys)
    df.to_csv(omic_dir + '/mean_std_corr.tsv', sep='\t', index=True)
        
# discriminative_score
for omic in omics:
    omic_dir = fdir + '/' + omic
    Path(omic_dir).mkdir(exist_ok=True, parents=True)
    keys = []
    dfs = []
    for dataset in datasets:
        print('dataset', dataset)
        for method in methods:
            print('method', method)
            method_result_dir = base_result_dir + '/' + method + '/' + dataset
            for subdir in method_dirs[method]:
                working_dir = method_result_dir + '/' + subdir + '/output'
                keys.append((dataset, method, subdir))
                if (method_hparam[method]):
                    best_hparam_path = open(working_dir + '/logs/csv_logs/best_hparam.txt', 'r')
                    best_hparam = best_hparam_path.readline().strip()
                    working_dir = working_dir + '/' + best_hparam
                working_dir = working_dir + '/evaluate/substitution/' + omic + '/fidelity'
                df = pd.read_csv(working_dir + '/discriminative_score.tsv', sep='\t', index_col='model', usecols=['model', 'auroc', 'auprc'])
                dfs.append(df)
    df = pd.concat(dfs, axis=0, keys=keys)
    df = df.reorder_levels(order=[3, 2, 0, 1], axis=0)
    df = df.sort_index(level=0, axis=0)
    df.to_csv(omic_dir + '/discriminative_score.tsv', sep='\t', index=True)
    
# correlation
cdir = sub_dir + '/correlation'
Path(cdir).mkdir(exist_ok=True, parents=True)

for omic in omics:
    omic_dir = cdir + '/' + omic
    Path(omic_dir).mkdir(exist_ok=True, parents=True)
    keys = []
    dfs = []
    for dataset in datasets:
        print('dataset', dataset)
        for method in methods:
            print('method', method)
            method_result_dir = base_result_dir + '/' + method + '/' + dataset
            for subdir in method_dirs[method]:
                working_dir = method_result_dir + '/' + subdir + '/output'
                keys.append((dataset, method, subdir))
                if (method_hparam[method]):
                    best_hparam_path = open(working_dir + '/logs/csv_logs/best_hparam.txt', 'r')
                    best_hparam = best_hparam_path.readline().strip()
                    working_dir = working_dir + '/' + best_hparam
                working_dir = working_dir + '/evaluate/substitution/' + omic + '/correlation'
                df = pd.read_csv(working_dir + '/correlation.tsv', sep='\t', index_col=0, header=None)
                dfs.append(df)
    df = pd.concat(dfs, axis=1, keys=keys)
    df = df.droplevel(level=3, axis=1)
    df.index.name = None
    df.to_csv(omic_dir + '/correlation.tsv', sep='\t', index=True)
        
# utility
udir = sub_dir + '/utility'
Path(udir).mkdir(exist_ok=True, parents=True)

for omic in omics:
    omic_dir = udir + '/' + omic
    Path(omic_dir).mkdir(exist_ok=True, parents=True)
    keys = []
    dfs = []
    for dataset in datasets:
        print('dataset', dataset)
        for method in methods:
            print('method', method)
            method_result_dir = base_result_dir + '/' + method + '/' + dataset
            for subdir in method_dirs[method]:
                working_dir = method_result_dir + '/' + subdir + '/output'
                keys.append((dataset, method, subdir))
                if (method_hparam[method]):
                    best_hparam_path = open(working_dir + '/logs/csv_logs/best_hparam.txt', 'r')
                    best_hparam = best_hparam_path.readline().strip()
                    working_dir = working_dir + '/' + best_hparam
                working_dir = working_dir + '/evaluate/substitution/' + omic + '/utility'
                df = pd.read_csv(working_dir + '/anova_pvalue.tsv', sep='\t', index_col=0)
                dfs.append(df)
    df = pd.concat(dfs, axis=1, keys=keys)
    df.to_csv(omic_dir + '/anova_pvalue.tsv', sep='\t', index=True)
        
for omic in omics:
    omic_dir = udir + '/' + omic
    Path(omic_dir).mkdir(exist_ok=True, parents=True)
    keys = []
    dfs = []
    for dataset in datasets:
        print('dataset', dataset)
        for method in methods:
            print('method', method)
            method_result_dir = base_result_dir + '/' + method + '/' + dataset
            for subdir in method_dirs[method]:
                working_dir = method_result_dir + '/' + subdir + '/output'
                keys.append((dataset, method, subdir))
                if (method_hparam[method]):
                    best_hparam_path = open(working_dir + '/logs/csv_logs/best_hparam.txt', 'r')
                    best_hparam = best_hparam_path.readline().strip()
                    working_dir = working_dir + '/' + best_hparam
                working_dir = working_dir + '/evaluate/substitution/' + omic + '/utility'
                df = pd.read_csv(working_dir + '/anova_pvalue_corr.tsv', sep='\t', index_col=0)
                dfs.append(df)
    df = pd.concat(dfs, axis=0, keys=keys)
    df.to_csv(omic_dir + '/anova_pvalue_corr.tsv', sep='\t', index=True)
        
for omic in omics:
    omic_dir = udir + '/' + omic
    Path(omic_dir).mkdir(exist_ok=True, parents=True)
    keys = []
    dfs = []
    for dataset in datasets:
        print('dataset', dataset)
        for method in methods:
            print('method', method)
            method_result_dir = base_result_dir + '/' + method + '/' + dataset
            for subdir in method_dirs[method]:
                working_dir = method_result_dir + '/' + subdir + '/output'
                keys.append((dataset, method, subdir))
                if (method_hparam[method]):
                    best_hparam_path = open(working_dir + '/logs/csv_logs/best_hparam.txt', 'r')
                    best_hparam = best_hparam_path.readline().strip()
                    working_dir = working_dir + '/' + best_hparam
                working_dir = working_dir + '/evaluate/substitution/' + omic + '/utility'
                df = pd.read_csv(working_dir + '/classification_utility.tsv', sep='\t', index_col='model')
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                dfs.append(df)
    df = pd.concat(dfs, axis=0, keys=keys)
    df.to_csv(omic_dir + '/classification_utility.tsv', sep='\t', index=True)

for omic in omics:
    omic_dir = udir + '/' + omic
    Path(omic_dir).mkdir(exist_ok=True, parents=True)
    keys = []
    dfs = []
    for dataset in datasets:
        print('dataset', dataset)
        for method in methods:
            print('method', method)
            method_result_dir = base_result_dir + '/' + method + '/' + dataset
            for subdir in method_dirs[method]:
                working_dir = method_result_dir + '/' + subdir + '/output'
                keys.append((dataset, method, subdir))
                if (method_hparam[method]):
                    best_hparam_path = open(working_dir + '/logs/csv_logs/best_hparam.txt', 'r')
                    best_hparam = best_hparam_path.readline().strip()
                    working_dir = working_dir + '/' + best_hparam
                working_dir = working_dir + '/evaluate/substitution/' + omic + '/utility'
                df = pd.read_csv(working_dir + '/kmeans_clustering.tsv', sep='\t', index_col='metric')
                dfs.append(df)
    df = pd.concat(dfs, axis=0, keys=keys)
    df.to_csv(omic_dir + '/kmeans_clustering.tsv', sep='\t', index=True)
        
# privacy
pdir = sub_dir + '/privacy'
Path(pdir).mkdir(exist_ok=True, parents=True)

for omic in omics:
    omic_dir = pdir + '/' + omic
    Path(omic_dir).mkdir(exist_ok=True, parents=True)
    keys = []
    dfs = []
    for dataset in datasets:
        print('dataset', dataset)
        for method in methods:
            print('method', method)
            method_result_dir = base_result_dir + '/' + method + '/' + dataset
            for subdir in method_dirs[method]:
                working_dir = method_result_dir + '/' + subdir + '/output'
                keys.append((dataset, method, subdir))
                if (method_hparam[method]):
                    best_hparam_path = open(working_dir + '/logs/csv_logs/best_hparam.txt', 'r')
                    best_hparam = best_hparam_path.readline().strip()
                    working_dir = working_dir + '/' + best_hparam
                working_dir = working_dir + '/evaluate/substitution/' + omic + '/privacy'
                df = pd.read_csv(working_dir + '/membership_inference.tsv', sep='\t', index_col=0, header=None)
                dfs.append(df)
    df = pd.concat(dfs, axis=0, keys=keys)
    df.to_csv(omic_dir + '/membership_inference.tsv', sep='\t', index=True)
        
for omic in omics:
    omic_dir = pdir + '/' + omic
    Path(omic_dir).mkdir(exist_ok=True, parents=True)
    keys = []
    dfs = []
    for dataset in datasets:
        print('dataset', dataset)
        for method in methods:
            print('method', method)
            method_result_dir = base_result_dir + '/' + method + '/' + dataset
            for subdir in method_dirs[method]:
                working_dir = method_result_dir + '/' + subdir + '/output'
                keys.append((dataset, method, subdir))
                if (method_hparam[method]):
                    best_hparam_path = open(working_dir + '/logs/csv_logs/best_hparam.txt', 'r')
                    best_hparam = best_hparam_path.readline().strip()
                    working_dir = working_dir + '/' + best_hparam
                working_dir = working_dir + '/evaluate/substitution/' + omic + '/privacy'
                df = pd.read_csv(working_dir + '/re_identification.tsv', sep='\t', index_col=0)
                dfs.append(df)
    df = pd.concat(dfs, axis=0, keys=keys)
    df.to_csv(omic_dir + '/re_identification.tsv', sep='\t', index=True)
        
# augmentation
aug_dir = summary_dir + '/augmentation'
Path(aug_dir).mkdir(exist_ok=True, parents=True)

for omic in omics:
    omic_dir = aug_dir + '/' + omic
    Path(omic_dir).mkdir(exist_ok=True, parents=True)
    keys = []
    dfs = []
    for dataset in datasets:
        print('dataset', dataset)
        for method in methods:
            print('method', method)
            method_result_dir = base_result_dir + '/' + method + '/' + dataset
            for subdir in method_dirs[method]:
                working_dir = method_result_dir + '/' + subdir + '/output'
                keys.append((dataset, method, subdir))
                if (method_hparam[method]):
                    best_hparam_path = open(working_dir + '/logs/csv_logs/best_hparam.txt', 'r')
                    best_hparam = best_hparam_path.readline().strip()
                    working_dir = working_dir + '/' + best_hparam
                working_dir = working_dir + '/evaluate/augmentation/' + omic
                df = pd.read_csv(working_dir + '/augmentated_classification.tsv', sep='\t', index_col=0)
                dfs.append(df)
    df = pd.concat(dfs, axis=0, keys=keys)
    df = df.reorder_levels([3, 2, 0, 1], axis=0)
    df.to_csv(omic_dir + '/augmentated_classification.tsv', sep='\t', index=True)