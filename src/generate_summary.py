import pandas as pd
import dataframe_image as dfi
from pathlib import Path
import os
import functools
print = functools.partial(print, flush=True)

summary_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Result/Summary'
#plot_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Result/Plot'

config_df = pd.read_csv('/shared/nas/data/m1/ksarker2/Synthetic/Result/PRADA/config.tsv',
                          sep='\t', usecols=['config_label', 'corr_top_k', 'prior', 'prior_top_k', 'model_name'])

data_config = config_df[config_df['prior'] == 'no-prior']

fdir = summary_dir + '/substitution/fidelity'
cdir = summary_dir + '/substitution/correlation'
udir = summary_dir + '/substitution/utility'
pdir = summary_dir + '/substitution/privacy'
adir = summary_dir + '/augmentation'

omics = ['mic', 'met', 'mic_met']

def generate_table_image(tsv_path, out_dir, max_cols, min_cols, all_max=False, all_min=False):
    print('tsv_path', tsv_path)
    
    full_df = pd.read_csv(tsv_path, sep='\t')
    print('full_df', full_df.shape)
    print(full_df['method'].value_counts())
    full_df = full_df.round(5)
    datasets = set(full_df.dataset)
    
    for dataset in datasets:
        print('dataset', dataset)
        df = full_df[full_df.dataset == dataset]
        print('df', df.shape)
        print(df['method'].value_counts())
        cols = set(df.columns)
        setting_cols = set(['dataset', 'method', 'config_label', 'corr_top_k', 'prior_top_k', 'model_name', 'prior'])
        setting_cols = list(cols.intersection(setting_cols))
        metric_cols = list(cols.difference(setting_cols))
        metric_cols.sort()
        setting_cols.sort()
        print('metric_cols', metric_cols)
        print('setting_cols', setting_cols)
        cols = setting_cols + metric_cols 
        df = df[cols]
        if(all_max):
            max_cols = metric_cols
        if(all_min):
            min_cols = metric_cols
        max_subset = pd.IndexSlice[df.index, max_cols]
        min_subset = pd.IndexSlice[df.index, min_cols]

        df_styled = df.style.format(precision=5).set_table_styles(
        [{"selector": "", "props": [("border", "1px solid grey"), ('text-align', 'center'), ('max-width', '100%'), ('white-space', 'nowrap')]},
          {"selector": "tbody td", "props": [("border", "1px solid grey"), ('text-align', 'center'), ('max-width', '100%'), ('white-space', 'nowrap')]},
         {"selector": "th", "props": [("border", "1px solid grey"), ('text-align', 'center'), ('max-width', '100%'), ('white-space', 'nowrap')]},
        ]).highlight_max(axis=0, props='font-weight: bold; color: blue', subset=max_subset).highlight_min(axis=0, props='font-weight: bold; color: blue', subset=min_subset)
        filename = os.path.basename(tsv_path).replace('.tsv', '.png')
        Path(out_dir + '/' + dataset).mkdir(parents=True, exist_ok=True)
        dfi.export(df_styled, out_dir + '/' + dataset + '/' + filename, table_conversion="matplotlib", max_rows=-1, max_cols=-1)

# fidelity
# kstest

# baseline comparison
for omic in omics:
    omic_dir = fdir + '/' + omic
    print('omic_dir', omic_dir)
    metric_df = pd.read_csv(omic_dir + '/kstest-pval.tsv', sep='\t', index_col=0, header=[0, 1, 2])
    print('metric_df', metric_df.shape)
    metric_df = metric_df.transpose().reset_index(names=['dataset', 'method', 'config_label']) 
    print('metric_df', metric_df.shape)
    
    metric_df = pd.merge(metric_df, data_config, on='config_label', how='left')
    print('metric_df', metric_df.shape)
    
    prada_df =  metric_df[metric_df.method == 'PRADA']
    print('prada_df', prada_df.shape)
    prada_df = prada_df.drop_duplicates(subset=['dataset', 'prior', 'method', 'model_name', 'corr_top_k'])
    print('prada_df', prada_df.shape)
    base_df =  metric_df[metric_df.method != 'PRADA']
    metric_df = pd.concat([prada_df, base_df], axis=0)
    print('metric_df', metric_df.shape)
    
    metric_df = metric_df[~((metric_df['method'] == 'PRADA') & (metric_df['prior'].isnull()))]
    print('metric_df', metric_df.shape)
    metric_df = metric_df.drop(columns='prior_top_k')
    
    tsv_path = omic_dir + '/kstest_pval_baseline_vs_prada.tsv'
    metric_df.to_csv(tsv_path, sep='\t', index=False)
    max_cols = ['mean', 'median', 'minimum', 'maximum']
    min_cols = []
    generate_table_image(tsv_path, omic_dir, max_cols, min_cols)
    
# data vs data+prior
for omic in omics:
    omic_dir = fdir + '/' + omic
    print(omic_dir)
    metric_df = pd.read_csv(omic_dir + '/kstest-pval.tsv', sep='\t', index_col=0, header=[0, 1, 2])
    metric_df = metric_df.transpose().reset_index(names=['dataset', 'method', 'config_label']) 
    
    metric_df = pd.merge(metric_df, config_df, on='config_label', how='inner')
    metric_df.loc[metric_df.prior == 'no-prior', 'prior_top_k'] = 0
    
    prada_df =  metric_df[metric_df.method == 'PRADA']
    prada_df = prada_df.drop_duplicates(subset=['dataset', 'prior', 'method', 'model_name', 'corr_top_k', 'prior_top_k'])
    base_df =  metric_df[metric_df.method != 'PRADA']
    metric_df = pd.concat([prada_df, base_df], axis=0)
    
    metric_df = metric_df.sort_values(by=['method', 'prior', 'model_name', 'corr_top_k', 'prior_top_k'])
    
    tsv_path = omic_dir + '/kstest_pval_data_vs_data+prior.tsv'
    metric_df.to_csv(tsv_path, sep='\t', index=False)
    max_cols = ['mean', 'median', 'minimum', 'maximum']
    min_cols = []
    generate_table_image(tsv_path, omic_dir, max_cols, min_cols)

# mean_std_corr

# baseline comparison
for omic in omics:
    omic_dir = fdir + '/' + omic
    print(omic_dir)
    metric_df = pd.read_csv(omic_dir + '/mean_std_corr.tsv', sep='\t', index_col=0, header=[0, 1, 2, 3])
    metric_df = metric_df.stack(level=3).transpose()
    metric_df.columns = metric_df.columns.map('_'.join).str.strip('_')
    metric_df = metric_df.reset_index(names=['dataset', 'method', 'config_label']) 
    print(metric_df.head())
    
    metric_df = pd.merge(metric_df, data_config, on='config_label', how='left')
    
    prada_df =  metric_df[metric_df.method == 'PRADA']
    prada_df = prada_df.drop_duplicates(subset=['dataset', 'prior', 'method', 'model_name', 'corr_top_k'])
    base_df =  metric_df[metric_df.method != 'PRADA']
    metric_df = pd.concat([prada_df, base_df], axis=0)
    
    metric_df = metric_df[~((metric_df['method'] == 'PRADA') & (metric_df['prior'].isnull()))]
    print('metric_df', metric_df.shape)
    print(metric_df)
    metric_df = metric_df.drop(columns='prior_top_k')
    
    tsv_path = omic_dir + '/mean_std_corr_pval_baseline_vs_prada.tsv'
    metric_df.to_csv(tsv_path, sep='\t', index=False)
    max_cols = ['mean_statistic', 'std_statistic']
    min_cols = ['mean_p-value', 'std_p-value']
    generate_table_image(tsv_path, omic_dir, max_cols, min_cols)
    
# data vs data+prior
for omic in omics:
    omic_dir = fdir + '/' + omic
    print(omic_dir)
    metric_df = pd.read_csv(omic_dir + '/mean_std_corr.tsv', sep='\t', index_col=0, header=[0, 1, 2, 3])
    metric_df = metric_df.stack(level=3).transpose()
    metric_df.columns = metric_df.columns.map('_'.join).str.strip('_')
    metric_df = metric_df.reset_index(names=['dataset', 'method', 'config_label'])
    
    print(metric_df.head())
    
    metric_df = pd.merge(metric_df, config_df, on='config_label', how='inner')
    metric_df.loc[metric_df.prior == 'no-prior', 'prior_top_k'] = 0
    
    prada_df =  metric_df[metric_df.method == 'PRADA']
    prada_df = prada_df.drop_duplicates(subset=['dataset', 'prior', 'method', 'model_name', 'corr_top_k', 'prior_top_k'])
    base_df =  metric_df[metric_df.method != 'PRADA']
    metric_df = pd.concat([prada_df, base_df], axis=0)
    
    metric_df = metric_df.sort_values(by=['method', 'prior', 'model_name', 'corr_top_k', 'prior_top_k'])
    metric_df.to_csv(omic_dir + '/mean_std_corr_data_vs_data+prior.tsv', sep='\t', index=False)
    
    tsv_path = omic_dir + '/mean_std_corr_data_vs_data+prior.tsv'
    metric_df.to_csv(tsv_path, sep='\t', index=False)
    max_cols = ['mean_statistic', 'std_statistic']
    min_cols = ['mean_p-value', 'std_p-value']
    generate_table_image(tsv_path, omic_dir, max_cols, min_cols)
    
# discriminative_score
print('discriminative_score')
# baseline comparison
for omic in omics:
    omic_dir = fdir + '/' + omic
    print(omic_dir)
    metric_df = pd.read_csv(omic_dir + '/discriminative_score.tsv', sep='\t', index_col=[0, 1, 2, 3])
    metric_df = metric_df.unstack(level=0)
    metric_df.columns = metric_df.columns.map('_'.join)
    
    
    metric_df = metric_df.reset_index(names=['config_label', 'dataset', 'method']) 
    metric_df = pd.merge(metric_df, data_config, on='config_label', how='left')
    print(metric_df)
    
    prada_df =  metric_df[metric_df.method == 'PRADA']
    prada_df = prada_df.drop_duplicates(subset=['dataset', 'prior', 'method', 'model_name', 'corr_top_k'])
    base_df =  metric_df[metric_df.method != 'PRADA']
    metric_df = pd.concat([prada_df, base_df], axis=0)
    
    metric_df = metric_df[~((metric_df['method'] == 'PRADA') & (metric_df['prior'].isnull()))]
    print('metric_df', metric_df.shape)
    print(metric_df)
    metric_df = metric_df.drop(columns='prior_top_k')
    metric_df.to_csv(omic_dir + '/discriminative_baseline_vs_prada.tsv', sep='\t', index=False)
    
    tsv_path = omic_dir + '/discriminative_baseline_vs_prada.tsv'
    metric_df.to_csv(tsv_path, sep='\t', index=False)
    generate_table_image(tsv_path, omic_dir, max_cols=[], min_cols=[], all_max=False, all_min=True)
    
# data vs data+prior
for omic in omics:
    omic_dir = fdir + '/' + omic
    print(omic_dir)
    metric_df = pd.read_csv(omic_dir + '/discriminative_score.tsv', sep='\t', index_col=[0, 1, 2, 3])
    metric_df = metric_df.unstack(level=0)
    metric_df.columns = metric_df.columns.map('_'.join)
    metric_df = metric_df.reset_index(names=['config_label', 'dataset', 'method']) 
    
    metric_df = pd.merge(metric_df, config_df, on='config_label', how='inner')
    metric_df.loc[metric_df.prior == 'no-prior', 'prior_top_k'] = 0
    
    prada_df =  metric_df[metric_df.method == 'PRADA']
    prada_df = prada_df.drop_duplicates(subset=['dataset', 'prior', 'method', 'model_name', 'corr_top_k', 'prior_top_k'])
    base_df =  metric_df[metric_df.method != 'PRADA']
    metric_df = pd.concat([prada_df, base_df], axis=0)
    
    metric_df = metric_df.sort_values(by=['prior', 'method', 'model_name', 'corr_top_k', 'prior_top_k'])
    
    tsv_path = omic_dir + '/discriminative_data_vs_data+prior.tsv'
    metric_df.to_csv(tsv_path, sep='\t', index=False)
    generate_table_image(tsv_path, omic_dir, max_cols=[], min_cols=[], all_max=False, all_min=True)
    
# correlation

# baseline comparison
for omic in omics:
    omic_dir = cdir + '/' + omic
    print(omic_dir)
    metric_df = pd.read_csv(omic_dir + '/correlation.tsv', sep='\t', index_col=0, header=[0, 1, 2])
    metric_df = metric_df.transpose().reset_index(names=['dataset', 'method', 'config_label']) 
    
    metric_df = pd.merge(metric_df, data_config, on='config_label', how='left')
    
    prada_df =  metric_df[metric_df.method == 'PRADA']
    prada_df = prada_df.drop_duplicates(subset=['dataset', 'prior', 'method', 'model_name', 'corr_top_k'])
    base_df =  metric_df[metric_df.method != 'PRADA']
    metric_df = pd.concat([prada_df, base_df], axis=0)
    
    metric_df = metric_df[~((metric_df['method'] == 'PRADA') & (metric_df['prior'].isnull()))]
    print('metric_df', metric_df.shape)
    print(metric_df)
    metric_df = metric_df.drop(columns='prior_top_k')
    metric_df.to_csv(omic_dir + '/correlation_baseline_vs_prada.tsv', sep='\t', index=False)
    
    tsv_path = omic_dir + '/correlation_baseline_vs_prada.tsv'
    metric_df.to_csv(tsv_path, sep='\t', index=False)
    max_cols = ['corr_accuracy', 'corr_corr']
    min_cols = ['avg_diff', 'corr_corr_pval']
    generate_table_image(tsv_path, omic_dir, max_cols, min_cols, all_max=False, all_min=False)
    
# data vs data+prior
for omic in omics:
    omic_dir = cdir + '/' + omic
    print(omic_dir)
    metric_df = pd.read_csv(omic_dir + '/correlation.tsv', sep='\t', index_col=0, header=[0, 1, 2])
    metric_df = metric_df.transpose().reset_index(names=['dataset', 'method', 'config_label']) 
    
    metric_df = pd.merge(metric_df, config_df, on='config_label', how='inner')
    metric_df.loc[metric_df.prior == 'no-prior', 'prior_top_k'] = 0
    
    prada_df =  metric_df[metric_df.method == 'PRADA']
    prada_df = prada_df.drop_duplicates(subset=['dataset', 'prior', 'method', 'model_name', 'corr_top_k', 'prior_top_k'])
    base_df =  metric_df[metric_df.method != 'PRADA']
    metric_df = pd.concat([prada_df, base_df], axis=0)
    
    metric_df = metric_df.sort_values(by=['prior', 'method', 'model_name', 'corr_top_k', 'prior_top_k'])
    metric_df.to_csv(omic_dir + '/correlation_data_vs_data+prior.tsv', sep='\t', index=False)
    
    tsv_path = omic_dir + '/correlation_data_vs_data+prior.tsv'
    metric_df.to_csv(tsv_path, sep='\t', index=False)
    max_cols = ['corr_accuracy', 'corr_corr']
    min_cols = ['avg_diff', 'corr_corr_pval']
    generate_table_image(tsv_path, omic_dir, max_cols, min_cols, all_max=False, all_min=False)
# utility
# classification

# baseline comparison
for omic in omics:
    omic_dir = udir + '/' + omic
    print(omic_dir)
    metric_df = pd.read_csv(omic_dir + '/classification_utility.tsv', sep='\t', index_col=[0, 1, 2, 3])
    metric_df = metric_df.unstack(level=3)
    metric_df.columns = metric_df.columns.map('_'.join)
    print(metric_df.head())
    #metric_df.columns = ['dataset', 'method', 'config', 'metric', 'value']
    metric_df = metric_df.reset_index(names=['dataset', 'method', 'config_label']) 
    
    print(metric_df.head())
    metric_df = pd.merge(metric_df, data_config, on='config_label', how='left')
    
    
    metric_df = metric_df[~((metric_df['method'] == 'PRADA') & (metric_df['prior'].isnull()))]
    print('metric_df', metric_df.shape)
    print(metric_df)
    metric_df = metric_df.drop(columns='prior_top_k')
    metric_df.to_csv(omic_dir + '/classification_baseline_vs_prada.tsv', sep='\t', index=False)
    
    tsv_path = omic_dir + '/classification_baseline_vs_prada.tsv'
    metric_df.to_csv(tsv_path, sep='\t', index=False)
    generate_table_image(tsv_path, omic_dir, max_cols=[], min_cols=[], all_max=True, all_min=False)
    
# data vs data+prior
for omic in omics:
    omic_dir = udir + '/' + omic
    print(omic_dir)
    metric_df = pd.read_csv(omic_dir + '/classification_utility.tsv', sep='\t', index_col=[0, 1, 2, 3])
    metric_df = metric_df.unstack(level=3)
    metric_df.columns = metric_df.columns.map('_'.join)
    print(metric_df.head())
    #metric_df.columns = ['dataset', 'method', 'config', 'metric', 'value']
    metric_df = metric_df.reset_index(names=['dataset', 'method', 'config_label']) 
    
    metric_df = pd.merge(metric_df, config_df, on='config_label', how='inner')
    metric_df.loc[metric_df.prior == 'no-prior', 'prior_top_k'] = 0
    
    prada_df =  metric_df[metric_df.method == 'PRADA']
    prada_df = prada_df.drop_duplicates(subset=['dataset', 'prior', 'method', 'model_name', 'corr_top_k', 'prior_top_k'])
    base_df =  metric_df[metric_df.method != 'PRADA']
    metric_df = pd.concat([prada_df, base_df], axis=0)
    
    metric_df = metric_df.sort_values(by=['prior', 'method', 'model_name', 'corr_top_k', 'prior_top_k'])
    metric_df.to_csv(omic_dir + '/classification_data_vs_data+prior.tsv', sep='\t', index=False)
    
    tsv_path = omic_dir + '/classification_data_vs_data+prior.tsv'
    metric_df.to_csv(tsv_path, sep='\t', index=False)
    generate_table_image(tsv_path, omic_dir, max_cols=[], min_cols=[], all_max=True, all_min=False)
    
# clustering

# baseline comparison
for omic in omics:
    omic_dir = udir + '/' + omic
    print(omic_dir)
    metric_df = pd.read_csv(omic_dir + '/kmeans_clustering.tsv', sep='\t', index_col=[0, 1, 2, 3])
    metric_df = metric_df.unstack(level=3)
    metric_df.columns = metric_df.columns.map('_'.join)
    print(metric_df.head())
    #metric_df.columns = ['dataset', 'method', 'config', 'metric', 'value']
    metric_df = metric_df.reset_index(names=['dataset', 'method', 'config_label']) 
    
    print(metric_df.head())
    metric_df = pd.merge(metric_df, data_config, on='config_label', how='left')
    prada_df =  metric_df[metric_df.method == 'PRADA']
    prada_df = prada_df.drop_duplicates(subset=['dataset', 'prior', 'method', 'model_name', 'corr_top_k'])
    base_df =  metric_df[metric_df.method != 'PRADA']
    metric_df = pd.concat([prada_df, base_df], axis=0)
    
    metric_df = metric_df[~((metric_df['method'] == 'PRADA') & (metric_df['prior'].isnull()))]
    print('metric_df', metric_df.shape)
    print(metric_df)
    metric_df = metric_df.drop(columns='prior_top_k')
    
    tsv_path = omic_dir + '/clustering_baseline_vs_prada.tsv'
    metric_df.to_csv(tsv_path, sep='\t', index=False)
    generate_table_image(tsv_path, omic_dir, max_cols=[], min_cols=[], all_max=True, all_min=False)
    
# data vs data+prior
for omic in omics:
    omic_dir = udir + '/' + omic
    print(omic_dir)
    metric_df = pd.read_csv(omic_dir + '/kmeans_clustering.tsv', sep='\t', index_col=[0, 1, 2, 3])
    metric_df = metric_df.unstack(level=3)
    metric_df.columns = metric_df.columns.map('_'.join)
    print(metric_df.head())
    #metric_df.columns = ['dataset', 'method', 'config', 'metric', 'value']
    metric_df = metric_df.reset_index(names=['dataset', 'method', 'config_label']) 
    
    metric_df = pd.merge(metric_df, config_df, on='config_label', how='inner')
    metric_df.loc[metric_df.prior == 'no-prior', 'prior_top_k'] = 0
    
    prada_df =  metric_df[metric_df.method == 'PRADA']
    prada_df = prada_df.drop_duplicates(subset=['dataset', 'prior', 'method', 'model_name', 'corr_top_k', 'prior_top_k'])
    base_df =  metric_df[metric_df.method != 'PRADA']
    metric_df = pd.concat([prada_df, base_df], axis=0)
    
    metric_df = metric_df.sort_values(by=['prior', 'method', 'model_name', 'corr_top_k', 'prior_top_k'])
    metric_df.to_csv(omic_dir + '/clustering_data_vs_data+prior.tsv', sep='\t', index=False)
    
    tsv_path = omic_dir + '/clustering_data_vs_data+prior.tsv'
    metric_df.to_csv(tsv_path, sep='\t', index=False)
    generate_table_image(tsv_path, omic_dir, max_cols=[], min_cols=[], all_max=True, all_min=False)
    
# anova

# baseline comparison
for omic in omics:
    omic_dir = udir + '/' + omic
    print(omic_dir)
    metric_df = pd.read_csv(omic_dir + '/anova_pvalue_corr.tsv', sep='\t', index_col=[0, 1, 2])
    metric_df = metric_df.reset_index(names=['dataset', 'method', 'config_label']) 
    print(metric_df.head())
    metric_df = pd.merge(metric_df, data_config, on='config_label', how='left')
    prada_df =  metric_df[metric_df.method == 'PRADA']
    prada_df = prada_df.drop_duplicates(subset=['dataset', 'prior', 'method', 'model_name', 'corr_top_k'])
    base_df =  metric_df[metric_df.method != 'PRADA']
    metric_df = pd.concat([prada_df, base_df], axis=0)
    
    metric_df = metric_df[~((metric_df['method'] == 'PRADA') & (metric_df['prior'].isnull()))]
    print('metric_df', metric_df.shape)
    print(metric_df)
    metric_df = metric_df.drop(columns='prior_top_k')
    metric_df.to_csv(omic_dir + '/anova_baseline_vs_prada.tsv', sep='\t', index=False)
    
    tsv_path = omic_dir + '/anova_baseline_vs_prada.tsv'
    metric_df.to_csv(tsv_path, sep='\t', index=False)
    generate_table_image(tsv_path, omic_dir, max_cols=['corr'], min_cols=['pvalue'], all_max=False, all_min=False)
    
# data vs data+prior
for omic in omics:
    omic_dir = udir + '/' + omic
    print(omic_dir)
    metric_df = pd.read_csv(omic_dir + '/anova_pvalue_corr.tsv', sep='\t', index_col=[0, 1, 2])
    metric_df = metric_df.reset_index(names=['dataset', 'method', 'config_label']) 
    
    metric_df = pd.merge(metric_df, config_df, on='config_label', how='inner')
    metric_df.loc[metric_df.prior == 'no-prior', 'prior_top_k'] = 0
    
    prada_df =  metric_df[metric_df.method == 'PRADA']
    prada_df = prada_df.drop_duplicates(subset=['dataset', 'prior', 'method', 'model_name', 'corr_top_k', 'prior_top_k'])
    base_df =  metric_df[metric_df.method != 'PRADA']
    metric_df = pd.concat([prada_df, base_df], axis=0)
    
    metric_df = metric_df.sort_values(by=['prior', 'method', 'model_name', 'corr_top_k', 'prior_top_k'])
    metric_df.to_csv(omic_dir + '/anova_data_vs_data+prior.tsv', sep='\t', index=False)
    
    tsv_path = omic_dir + '/anova_data_vs_data+prior.tsv'
    metric_df.to_csv(tsv_path, sep='\t', index=False)
    generate_table_image(tsv_path, omic_dir, max_cols=['corr'], min_cols=['pvalue'], all_max=False, all_min=False)
    

# privacy
# membership

# baseline comparison
for omic in omics:
    omic_dir = pdir + '/' + omic
    print(omic_dir)
    metric_df = pd.read_csv(omic_dir + '/membership_inference.tsv', sep='\t', index_col=[0, 1, 2, 3])
    metric_df = metric_df.unstack(level=3)
    metric_df = metric_df.droplevel(level=0, axis=1)
    print(metric_df.head())
    #metric_df.columns = ['dataset', 'method', 'config', 'metric', 'value']
    metric_df = metric_df.reset_index(names=['dataset', 'method', 'config_label']) 
    
    print(metric_df.head())
    metric_df = pd.merge(metric_df, data_config, on='config_label', how='left')
    prada_df =  metric_df[metric_df.method == 'PRADA']
    prada_df = prada_df.drop_duplicates(subset=['dataset', 'prior', 'method', 'model_name', 'corr_top_k'])
    base_df =  metric_df[metric_df.method != 'PRADA']
    metric_df = pd.concat([prada_df, base_df], axis=0)
    
    metric_df = metric_df[~((metric_df['method'] == 'PRADA') & (metric_df['prior'].isnull()))]
    print('metric_df', metric_df.shape)
    print(metric_df)
    metric_df = metric_df.drop(columns='prior_top_k')
    metric_df.to_csv(omic_dir + '/membership_baseline_vs_prada.tsv', sep='\t', index=False)
    
    tsv_path = omic_dir + '/membership_baseline_vs_prada.tsv'
    metric_df.to_csv(tsv_path, sep='\t', index=False)
    generate_table_image(tsv_path, omic_dir, max_cols=[], min_cols=[], all_max=True, all_min=False)
    
# data vs data+prior
for omic in omics:
    omic_dir = pdir + '/' + omic
    print(omic_dir)
    metric_df = pd.read_csv(omic_dir + '/membership_inference.tsv', sep='\t', index_col=[0, 1, 2, 3])
    metric_df = metric_df.unstack(level=3)
    metric_df = metric_df.droplevel(level=0, axis=1)
    print(metric_df.head())
    #metric_df.columns = ['dataset', 'method', 'config', 'metric', 'value']
    metric_df = metric_df.reset_index(names=['dataset', 'method', 'config_label']) 
    
    metric_df = pd.merge(metric_df, config_df, on='config_label', how='inner')
    metric_df.loc[metric_df.prior == 'no-prior', 'prior_top_k'] = 0
    
    prada_df =  metric_df[metric_df.method == 'PRADA']
    prada_df = prada_df.drop_duplicates(subset=['dataset', 'prior', 'method', 'model_name', 'corr_top_k', 'prior_top_k'])
    base_df =  metric_df[metric_df.method != 'PRADA']
    metric_df = pd.concat([prada_df, base_df], axis=0)
    
    metric_df = metric_df.sort_values(by=['prior', 'method', 'model_name', 'corr_top_k', 'prior_top_k'])
    
    tsv_path = omic_dir + '/membership_data_vs_data+prior.tsv'
    metric_df.to_csv(tsv_path, sep='\t', index=False)
    generate_table_image(tsv_path, omic_dir, max_cols=[], min_cols=[], all_max=True, all_min=False)
    
# re_identification

# baseline comparison
for omic in omics:
    omic_dir = pdir + '/' + omic
    print(omic_dir)
    metric_df = pd.read_csv(omic_dir + '/re_identification.tsv', sep='\t', index_col=[0, 1, 2, 3])
    metric_df = metric_df.unstack(level=3)
    metric_df.columns = metric_df.columns.map('_'.join)
    print(metric_df.head())
    #metric_df.columns = ['dataset', 'method', 'config', 'metric', 'value']
    metric_df = metric_df.reset_index(names=['dataset', 'method', 'config_label']) 
    
    print(metric_df.head())
    metric_df = pd.merge(metric_df, data_config, on='config_label', how='left')
    prada_df =  metric_df[metric_df.method == 'PRADA']
    prada_df = prada_df.drop_duplicates(subset=['dataset', 'prior', 'method', 'model_name', 'corr_top_k'])
    base_df =  metric_df[metric_df.method != 'PRADA']
    metric_df = pd.concat([prada_df, base_df], axis=0)
    
    metric_df = metric_df[~((metric_df['method'] == 'PRADA') & (metric_df['prior'].isnull()))]
    print('metric_df', metric_df.shape)
    print(metric_df)
    metric_df = metric_df.drop(columns='prior_top_k')
    metric_df.to_csv(omic_dir + '/re_identification_baseline_vs_prada.tsv', sep='\t', index=False)
    
    tsv_path = omic_dir + '/re_identification_baseline_vs_prada.tsv'
    metric_df.to_csv(tsv_path, sep='\t', index=False)
    generate_table_image(tsv_path, omic_dir, max_cols=[], min_cols=[], all_max=True, all_min=False)
    
# data vs data+prior
for omic in omics:
    omic_dir = pdir + '/' + omic
    print(omic_dir)
    metric_df = pd.read_csv(omic_dir + '/re_identification.tsv', sep='\t', index_col=[0, 1, 2, 3])
    metric_df = metric_df.unstack(level=3)
    metric_df.columns = metric_df.columns.map('_'.join)
    print(metric_df.head())
    #metric_df.columns = ['dataset', 'method', 'config', 'metric', 'value']
    metric_df = metric_df.reset_index(names=['dataset', 'method', 'config_label']) 
    
    metric_df = pd.merge(metric_df, config_df, on='config_label', how='inner')
    metric_df.loc[metric_df.prior == 'no-prior', 'prior_top_k'] = 0
    
    prada_df =  metric_df[metric_df.method == 'PRADA']
    prada_df = prada_df.drop_duplicates(subset=['dataset', 'prior', 'method', 'model_name', 'corr_top_k', 'prior_top_k'])
    base_df =  metric_df[metric_df.method != 'PRADA']
    metric_df = pd.concat([prada_df, base_df], axis=0)
    
    metric_df = metric_df.sort_values(by=['prior', 'method', 'model_name', 'corr_top_k', 'prior_top_k'])
    metric_df.to_csv(omic_dir + '/re_identification_data_vs_data+prior.tsv', sep='\t', index=False)
    
    tsv_path = omic_dir + '/re_identification_data_vs_data+prior.tsv'
    metric_df.to_csv(tsv_path, sep='\t', index=False)
    generate_table_image(tsv_path, omic_dir, max_cols=[], min_cols=[], all_max=True, all_min=False)
    
# augmentation
# classification

# baseline comparison
for omic in omics:
    omic_dir = adir + '/' + omic
    print(omic_dir)
    metric_df = pd.read_csv(omic_dir + '/augmentated_classification.tsv', sep='\t', index_col=[0, 1, 2, 3])
    metric_df['auroc'] = (metric_df['aug_auroc'] > metric_df['exp_auroc']).astype(int)
    metric_df['auprc'] = (metric_df['aug_auprc'] > metric_df['exp_auprc']).astype(int)
    metric_df = metric_df.drop(columns=['exp_auroc', 'aug_auroc', 'exp_auprc', 'aug_auprc'])
    metric_df = metric_df.unstack(level=0)
    metric_df.columns = metric_df.columns.map('_'.join)
    print(metric_df.head())
    
    #metric_df.columns = ['dataset', 'method', 'config', 'metric', 'value']
    metric_df = metric_df.reset_index(names=['config_label', 'dataset', 'method']) 
    
    print(metric_df.head())
    metric_df = pd.merge(metric_df, data_config, on='config_label', how='left')
    
    prada_df =  metric_df[metric_df.method == 'PRADA']
    prada_df = prada_df.drop_duplicates(subset=['dataset', 'prior', 'method', 'model_name', 'corr_top_k'])
    base_df =  metric_df[metric_df.method != 'PRADA']
    metric_df = pd.concat([prada_df, base_df], axis=0)
    
    metric_df = metric_df[~((metric_df['method'] == 'PRADA') & (metric_df['prior'].isnull()))]
    print('metric_df', metric_df.shape)
    print(metric_df)
    metric_df = metric_df.drop(columns='prior_top_k')
    metric_df.to_csv(omic_dir + '/classification_baseline_vs_prada.tsv', sep='\t', index=False)
    
    tsv_path = omic_dir + '/classification_baseline_vs_prada.tsv'
    metric_df.to_csv(tsv_path, sep='\t', index=False)
    generate_table_image(tsv_path, omic_dir, max_cols=[], min_cols=[], all_max=True, all_min=False)
    
# data vs data+prior
for omic in omics:
    omic_dir = adir + '/' + omic
    print(omic_dir)
    metric_df = pd.read_csv(omic_dir + '/augmentated_classification.tsv', sep='\t', index_col=[0, 1, 2, 3])
    metric_df['auroc'] = (metric_df['aug_auroc'] > metric_df['exp_auroc']).astype(int)
    metric_df['auprc'] = (metric_df['aug_auprc'] > metric_df['exp_auprc']).astype(int)
    metric_df = metric_df.drop(columns=['exp_auroc', 'aug_auroc', 'exp_auprc', 'aug_auprc'])
    metric_df = metric_df.unstack(level=0)
    metric_df.columns = metric_df.columns.map('_'.join)
    print(metric_df.head())
    #metric_df.columns = ['dataset', 'method', 'config', 'metric', 'value']
    metric_df = metric_df.reset_index(names=['config_label', 'dataset', 'method']) 
    
    metric_df = pd.merge(metric_df, config_df, on='config_label', how='inner')
    metric_df.loc[metric_df.prior == 'no-prior', 'prior_top_k'] = 0
    
    prada_df =  metric_df[metric_df.method == 'PRADA']
    prada_df = prada_df.drop_duplicates(subset=['dataset', 'prior', 'method', 'model_name', 'corr_top_k', 'prior_top_k'])
    base_df =  metric_df[metric_df.method != 'PRADA']
    metric_df = pd.concat([prada_df, base_df], axis=0)
    
    metric_df = metric_df.sort_values(by=['prior', 'method', 'model_name', 'corr_top_k', 'prior_top_k'])
    metric_df.to_csv(omic_dir + '/classification_data_vs_data+prior.tsv', sep='\t', index=False)
    
    tsv_path = omic_dir + '/classification_data_vs_data+prior.tsv'
    metric_df.to_csv(tsv_path, sep='\t', index=False)
    generate_table_image(tsv_path, omic_dir, max_cols=[], min_cols=[], all_max=True, all_min=False)