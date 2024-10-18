from pathlib import Path
import pandas as pd

summary_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Manuscript/corr-all/Summary'
base_result_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Manuscript/corr-all'

methods = ['GCVAE', 'CVAE']

method_dir_map = {
    10: {
        'GCVAE': ['config-' + str(i) for i in range(1, 13)],
        'CVAE': ['config-1']
    },
    50: {
        'GCVAE': ['config-' + str(i) for i in range(13, 25)],
        'CVAE': ['config-13']
    },
    100:
    {
        'GCVAE': ['config-' + str(i) for i in range(25, 37)],
        'CVAE': ['config-25']
    }
}

method_hparam = {
        'GCVAE': True,
        'CVAE': True
}

datasets = ['ST001386']

# substitution
sub_dir = summary_dir + '/substitution'
Path(sub_dir).mkdir(exist_ok=True, parents=True)

# kstest
      
# fidelity
fdir = sub_dir + '/fidelity'
Path(fdir).mkdir(exist_ok=True, parents=True)

# kstest

for feature_count, method_dirs in method_dir_map.items():
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
                working_dir = working_dir + '/evaluate/substitution/fidelity'
                df = pd.read_csv(working_dir + '/kstest.tsv', sep='\t', index_col='column')
                dfs.append(df)
    df = pd.concat(dfs, axis=1, keys=keys)
    df.to_csv(fdir + '/' + str(feature_count) + '.kstest.tsv', sep='\t', index=True)
        
# kstest-pval
for feature_count, method_dirs in method_dir_map.items():
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
                working_dir = working_dir + '/evaluate/substitution/fidelity'
                df = pd.read_csv(working_dir + '/kstest-pval.tsv', sep='\t', index_col=0, header=None)

                dfs.append(df)
    df = pd.concat(dfs, axis=1, keys=keys)
    df = df.droplevel(level=3, axis=1)
    df.index.name = None
    df.to_csv(fdir + '/' + str(feature_count) + '.kstest-pval.tsv', sep='\t', index=True)
        
# mean_std_corr
for feature_count, method_dirs in method_dir_map.items():
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
                working_dir = working_dir + '/evaluate/substitution/fidelity'
                df = pd.read_csv(working_dir + '/mean_std_corr.tsv', sep='\t', index_col=0)
                dfs.append(df)
    df = pd.concat(dfs, axis=1, keys=keys)
    df.to_csv(fdir + '/' + str(feature_count) + '.mean_std_corr.tsv', sep='\t', index=True)
        
# discriminative_score
for feature_count, method_dirs in method_dir_map.items():
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
                working_dir = working_dir + '/evaluate/substitution/fidelity'
                df = pd.read_csv(working_dir + '/discriminative_score.tsv', sep='\t', index_col='model', usecols=['model', 'auroc', 'auprc'])
                dfs.append(df)
    df = pd.concat(dfs, axis=0, keys=keys)
    df = df.reorder_levels(order=[3, 2, 0, 1], axis=0)
    df = df.sort_index(level=0, axis=0)
    df.to_csv(fdir + '/' + str(feature_count) + '.discriminative_score.tsv', sep='\t', index=True)
    
# correlation
cdir = sub_dir + '/correlation'
Path(cdir).mkdir(exist_ok=True, parents=True)

for feature_count, method_dirs in method_dir_map.items():
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
                working_dir = working_dir + '/evaluate/substitution/correlation'
                df = pd.read_csv(working_dir + '/correlation.tsv', sep='\t', index_col=0, header=None)
                dfs.append(df)
    df = pd.concat(dfs, axis=1, keys=keys)
    df = df.droplevel(level=3, axis=1)
    df.index.name = None
    df.to_csv(cdir + '/' + str(feature_count) + '.correlation.tsv', sep='\t', index=True)
        
# utility
udir = sub_dir + '/utility'
Path(udir).mkdir(exist_ok=True, parents=True)

for feature_count, method_dirs in method_dir_map.items():
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
                working_dir = working_dir + '/evaluate/substitution/utility'
                df = pd.read_csv(working_dir + '/anova_pvalue.tsv', sep='\t', index_col=0)
                dfs.append(df)
    df = pd.concat(dfs, axis=1, keys=keys)
    df.to_csv(udir + '/' + str(feature_count) + '.anova_pvalue.tsv', sep='\t', index=True)
        
for feature_count, method_dirs in method_dir_map.items():
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
                working_dir = working_dir + '/evaluate/substitution/utility'
                df = pd.read_csv(working_dir + '/anova_pvalue_corr.tsv', sep='\t', index_col=0)
                dfs.append(df)
    df = pd.concat(dfs, axis=0, keys=keys)
    df.to_csv(udir + '/' + str(feature_count) + '.anova_pvalue_corr.tsv', sep='\t', index=True)
        
for feature_count, method_dirs in method_dir_map.items():
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
                working_dir = working_dir + '/evaluate/substitution/utility'
                df = pd.read_csv(working_dir + '/classification_utility.tsv', sep='\t', index_col='model')
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                dfs.append(df)
    df = pd.concat(dfs, axis=0, keys=keys)
    df.to_csv(udir + '/' + str(feature_count) + '.classification_utility.tsv', sep='\t', index=True)

for feature_count, method_dirs in method_dir_map.items():
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
                working_dir = working_dir + '/evaluate/substitution/utility'
                df = pd.read_csv(working_dir + '/kmeans_clustering.tsv', sep='\t', index_col='metric')
                dfs.append(df)
    df = pd.concat(dfs, axis=0, keys=keys)
    df.to_csv(udir + '/' + str(feature_count) + '.kmeans_clustering.tsv', sep='\t', index=True)
        
# privacy
pdir = sub_dir + '/privacy'
Path(pdir).mkdir(exist_ok=True, parents=True)

for feature_count, method_dirs in method_dir_map.items():
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
                working_dir = working_dir + '/evaluate/substitution/privacy'
                df = pd.read_csv(working_dir + '/membership_inference.tsv', sep='\t', index_col=0, header=None)
                dfs.append(df)
    df = pd.concat(dfs, axis=0, keys=keys)
    df.to_csv(pdir + '/' + str(feature_count) + '.membership_inference.tsv', sep='\t', index=True)
        
for feature_count, method_dirs in method_dir_map.items():
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
                working_dir = working_dir + '/evaluate/substitution/privacy'
                df = pd.read_csv(working_dir + '/re_identification.tsv', sep='\t', index_col=0)
                dfs.append(df)
    df = pd.concat(dfs, axis=0, keys=keys)
    df.to_csv(pdir + '/' + str(feature_count) + '.re_identification.tsv', sep='\t', index=True)
        
# augmentation
aug_dir = summary_dir + '/augmentation'
Path(aug_dir).mkdir(exist_ok=True, parents=True)

for feature_count, method_dirs in method_dir_map.items():
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
                working_dir = working_dir + '/evaluate/augmentation'
                df = pd.read_csv(working_dir + '/augmentated_classification.tsv', sep='\t', index_col=0)
                dfs.append(df)
    df = pd.concat(dfs, axis=0, keys=keys)
    df = df.reorder_levels([3, 2, 0, 1], axis=0)
    df.to_csv(aug_dir + '/' + str(feature_count) + '.augmentated_classification.tsv', sep='\t', index=True)