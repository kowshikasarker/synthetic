import pandas as pd
import numpy as np
from pathlib import Path
import dataframe_image as dfi
import warnings
warnings.filterwarnings("ignore")

summary_dir = '/shared/nas/data/m1/ksarker2/RECOMB/Summary/ST001386'
fdir = summary_dir + '/substitution/fidelity'
cdir = summary_dir + '/substitution/correlation'
udir = summary_dir + '/substitution/utility'
pdir = summary_dir + '/substitution/privacy'

plot_dir = summary_dir + '/plot' 

# cvae vs gcvae
out_dir = plot_dir + '/gcvase_vs_cvae'
Path(out_dir).mkdir(parents=True, exist_ok=True)

feature_counts = [10, 40, 70, 100]
corr_methods = ['sp', 'pr', 'dcov', 'dcol', 'all']

config_df = pd.read_csv('/shared/nas/data/m1/ksarker2/RECOMB/Result/ST001386/config.tsv', sep='\t')
config_df['corr_method'] = config_df['corr_method'].str.replace('sp pr dcov dcol', 'all')
config_df = config_df[config_df.prior_top_pct == 0]
config_df = config_df[['config_label', 'corr_method', 'corr_top_pct']]
prior_zero_configs = list(config_df['config_label'])
config_df.columns = pd.MultiIndex.from_tuples([('config_label', '', ''),
                                               ('corr_method', '', ''),
                                               ('corr_top_pct', '', '')])
all_dfs = {}
better_direction = np.array([-1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
max_mask = (better_direction > 0).astype(int)
print('max_mask', max_mask)
min_mask = (better_direction < 0).astype(int)
print('min_mask', min_mask)
for feature_count in feature_counts:
    dfs = []
    
    df_path = fdir + '/' + str(feature_count) + '.kstest.tsv'
    df = pd.read_csv(df_path, sep='\t', index_col=0, header=[0, 1, 2, 3])
    df = df.median(axis=0)
    df = df.unstack(level=3)
    df = df.droplevel(level=0, axis=0)
    df.columns = pd.MultiIndex.from_tuples([('fidelity', 'kstest', 'pval▲'), ('fidelity', 'kstest', 'dist▼')])
    dfs.append(df)
    
    df_path = fdir + '/' + str(feature_count) + '.mean_std_corr.tsv'
    df = pd.read_csv(df_path, sep='\t', index_col=0, header=[0, 1, 2, 3])
    df = df.droplevel(level=0, axis=1)
    df = df.abs()
    df = df.rename(columns={'statistic': '|corr|▲', 'p-value': 'pval▼'}, level=2)
    df = df.stack(level=[0, 1])
    df = df.unstack(level=0)
    df = df.reorder_levels([1, 0], axis=1)
    df = df.sort_index(axis=1, level=0)
    df = pd.concat({'fidelity': df}, axis=1)
    dfs.append(df)
    
    df_path = fdir + '/' + str(feature_count) + '.discriminative_score.tsv'
    df = pd.read_csv(df_path, sep='\t', index_col=[0, 1, 2, 3])
    df.columns = df.columns.astype(str) + '▼'
    df = df.unstack(level=0)
    df = df.droplevel(level=1, axis=0)
    df = df.reorder_levels([1, 0], axis=0)
    df = pd.concat({'fidelity': df}, axis=1)
    dfs.append(df)
    
    print('Correlation', end='\n')
    
    df_path = cdir + '/' + str(feature_count) + '.correlation.tsv'
    df = pd.read_csv(df_path, sep='\t', index_col=0, header=[0, 1, 2])
    df = df.droplevel(level=0, axis=1)
    df = df.transpose()
    df.columns = pd.MultiIndex.from_tuples([('correlation', 'diff', 'mean▼'), ('correlation' , 'bin', 'acc▲'), ('correlation', 'sp', '|corr|▲'), ('correlation', 'sp', 'pval▼')])
    dfs.append(df)
    
    df_path = udir + '/' + str(feature_count) + '.anova_pvalue_corr.tsv'
    df = pd.read_csv(df_path, sep='\t', index_col=[0, 1, 2])
    df = df.droplevel(level=0, axis=0)
    df = df.drop(columns='pvalue')
    print(df.shape)
    df.columns = pd.MultiIndex.from_tuples([('utility', 'anova', '|corr|▲')])
    df = df.abs()
    dfs.append(df)
    
    df_path = udir + '/' + str(feature_count) + '.kmeans_clustering.tsv'
    df = pd.read_csv(df_path, sep='\t', index_col=[0, 1, 2, 3])
    df = df.droplevel(level=0, axis=0)
    df = df.drop(columns='exp')
    df = df.unstack(level=2)    
    df.columns = pd.MultiIndex.from_tuples([('utility', 'kmeans', 'adj_rand▲'), ('utility', 'kmeans', 'silhouette▲')])
    dfs.append(df)
    
    df_path = udir + '/' + str(feature_count) + '.classification_utility.tsv'
    df = pd.read_csv(df_path, sep='\t', index_col=[0, 1, 2, 3])
    df = df.loc[pd.IndexSlice[:, :, :, 'mlp'], :]
    df = df.droplevel(level=[0, 3], axis=0)
    df = df[['tstr_auroc', 'trstr_auroc', 'tsrtr_auroc', 'tstr_auprc', 'trstr_auprc', 'tsrtr_auprc']]
    df.columns = pd.MultiIndex.from_tuples([('utility', 'auroc', 'tstr▲'), ('utility', 'auroc', 'trstr▲'), ('utility', 'auroc', 'tsrtr▲'),
                                            ('utility', 'auprc', 'tstr▲'), ('utility', 'auprc', 'trstr▲'), ('utility', 'auprc', 'tsrtr▲')])
    dfs.append(df)
    
    df_path = pdir + '/' + str(feature_count) + '.membership_inference.tsv'
    df = pd.read_csv(df_path, sep='\t', index_col=[0, 1, 2, 3])
    df = df.droplevel(level=0, axis=0)
    df = df.unstack(level=2)    
    df.columns = pd.MultiIndex.from_tuples([('privacy', 'membership', 'auroc▲'), ('privacy', 'membership', 'auprc▲')])
    
    
    #df.to_csv(out_dir + '/plot.tsv', sep='\t', index=True)
    dfs.append(df)
    
    df = pd.concat(dfs, axis=1)

    df = df.reset_index()
    df = df.rename(columns={'level_0': 'method', 'level_1': 'config_label'})
    df = df[df.config_label.isin(prior_zero_configs)]
    #df = df.join(config_df, how='inner', on='config_label')
    df = pd.merge(df, config_df, on=[('config_label', '', '')], how='inner')
    df = df.drop(columns=['config_label', ('config_label', '', '')])
    columns = list(df.columns[:-2]) + [('edge', 'corr', 'type'), ('edge', 'corr', 'pct')]
    print(columns)
    df.columns = pd.MultiIndex.from_tuples(columns)
    last_cols = df.iloc[:, -2:]  # extract last 2 columns
    df = pd.concat([last_cols, df.iloc[:, :-2]], axis=1)
    df.loc[df[('method', '', '')] == 'cvae', [('edge', 'corr', 'type'), ('edge', 'corr', 'pct')]] = 'BASELINE'
    df = df.drop(columns=[('method', '', '')])
    df = df.sort_values(by=[('edge', 'corr', 'type'), ('edge', 'corr', 'pct')])
    df.to_csv(out_dir + '/' + str(feature_count) + '.plot.tsv', sep='\t', index=False)
    
    df = df.drop(columns=[('fidelity', 'kstest', 'pval▲'),
                     ('fidelity', 'mean', 'pval▼'),
                     ('fidelity', 'std', 'pval▼'),
                     ('correlation', 'sp', 'pval▼')])
    df = df.round(2)
    all_dfs[feature_count]= df
    gb = df.groupby(by=[('edge', 'corr', 'type')])
    
    baseline = df.loc[df[('edge', 'corr', 'pct')] == 'BASELINE', df.columns[2:]] 
    
    def color_cells(val):
      if val == 0:
          color = 'khaki'
      elif val < 0:
          color = 'mistyrose'
      else:
          color = 'paleturquoise'
      return f'background-color: {color}'
    
    def highlight_baseline(val):
        if val == 'BASELINE':
            return 'background-color: #893101; color: white; font-weight: bold'
        else:
            return 'table.border-color: red; table-border-width: 5 px'
    
    cols = np.array(df.columns[2:])
    assert better_direction.size == cols.size
    
    max_cols = cols[np.where(better_direction > 0)[0]]
    min_cols = cols[np.where(better_direction < 0)[0]]
    for corr_method, corr_df in gb:
        print('corr_method', corr_method)
        sub_df = corr_df.drop(columns=[('edge', 'corr', 'type')])
        sub_df = pd.concat([sub_df, df.loc[df[('edge', 'corr', 'pct')] == 'BASELINE', df.columns[1:]]], axis=0).reset_index(drop=True)
        print(sub_df.shape)
        df1 = sub_df[list(sub_df.columns[1:])]
        df2 = baseline.loc[baseline.index.repeat(df1.shape[0])].reset_index(drop=True)
        diff = df1.subtract(df2)
        diff = diff.mul(better_direction, axis='columns')
        print(df1)
        print(df2)
        print(diff)
        idx = pd.IndexSlice[sub_df[('edge', 'corr', 'pct')] != 'BASELINE', sub_df.columns[1:]]
        df_styled = sub_df.style.hide(axis='index').format(precision=2).set_table_styles(
          [{"selector": "", "props": [("border", "1px solid grey"), ('text-align', 'center'), ('max-width', '100%'), ('white-space', 'nowrap')]},
            {"selector": "tbody td", "props": [("border", "1px solid grey"), ('text-align', 'center'), ('max-width', '100%'), ('white-space', 'nowrap')]},
          {"selector": "th", "props": [("border", "1px solid black"), ('text-align', 'center'), ('max-width', '100%'), ('white-space', 'nowrap')]},
          ]).set_properties(**{'max-width': '100%', 'white-space': 'nowrap'}).highlight_max(axis=0, props='font-weight: bold; color: blue', subset=idx).apply(lambda x: diff.applymap(color_cells), axis=None, subset=pd.IndexSlice[:, sub_df.columns[1:]]).applymap(highlight_baseline, subset=pd.IndexSlice[sub_df[('edge', 'corr', 'pct')] == 'BASELINE', :])
        dfi.export(df_styled, out_dir + '/' + str(feature_count) + '.' + corr_method[0] + '.substitution.png', table_conversion='selenium')
    
    
    
    corr_method_max = gb[df.columns[2:]].max()
    corr_method_min = gb[df.columns[2:]].min()
    corr_method_best = corr_method_max.multiply(max_mask, axis='columns') + corr_method_min.multiply(min_mask, axis='columns')
    corr_method_best = corr_method_best.reset_index()
    corr_method_best['sort_col'] = corr_method_best[(('edge', 'corr', 'type'))] != 'BASELINE'
    corr_method_best = corr_method_best.sort_values('sort_col', ascending=False).drop('sort_col', axis=1)
    corr_method_best.to_csv(out_dir + '/' + str(feature_count) + '.corr_method_best.tsv', sep='\t', index=False)
    corr_method_best_diff = corr_method_best[corr_method_best.columns[1:]].subtract(baseline.to_numpy())
    corr_method_best_diff = corr_method_best_diff.mul(better_direction, axis='columns')
    print(corr_method_best_diff)
    corr_method_best_styled = corr_method_best.style.hide(axis='index').format(precision=2).set_table_styles(
          [{"selector": "", "props": [("border", "1px solid grey"), ('text-align', 'center'), ('max-width', '100%'), ('white-space', 'nowrap')]},
            {"selector": "tbody td", "props": [("border", "1px solid grey"), ('text-align', 'center'), ('max-width', '100%'), ('white-space', 'nowrap')]},
          {"selector": "th", "props": [("border", "1px solid black"), ('text-align', 'center'), ('max-width', '100%'), ('white-space', 'nowrap')]},
          ]).set_properties(**{'max-width': '100%', 'white-space': 'nowrap'}).highlight_max(axis=0, props='font-weight: bold; color: blue', subset=pd.IndexSlice[:, max_cols]).highlight_min(axis=0, props='font-weight: bold; color: blue', subset=pd.IndexSlice[:, min_cols]).apply(lambda x: corr_method_best_diff.applymap(color_cells), axis=None, subset=pd.IndexSlice[:, corr_method_best.columns[1:]]).applymap(highlight_baseline, subset=pd.IndexSlice[corr_method_best[('edge', 'corr', 'type')] == 'BASELINE', :])
    dfi.export(corr_method_best_styled, out_dir + '/' + str(feature_count) + '.corr_method_best.png', table_conversion='selenium')
    
    gb = df.groupby([('edge', 'corr', 'pct')])
    corr_pct_max = gb[df.columns[2:]].max()
    corr_pct_min = gb[df.columns[2:]].min()
    corr_pct_best = corr_pct_max.multiply(max_mask, axis='columns') + corr_pct_min.multiply(min_mask, axis='columns')
    corr_pct_best = corr_pct_best.reset_index()
    corr_pct_best['sort_col'] = corr_pct_best[(('edge', 'corr', 'pct'))] != 'BASELINE'
    corr_pct_best = corr_pct_best.sort_values('sort_col', ascending=False).drop('sort_col', axis=1)
    corr_pct_best.to_csv(out_dir + '/' + str(feature_count) + '.corr_pct_best.tsv', sep='\t', index=False)
    corr_pct_best_diff = corr_pct_best[corr_pct_best.columns[1:]].subtract(baseline.to_numpy())
    corr_pct_best_diff = corr_pct_best_diff.mul(better_direction, axis='columns')
    print(corr_pct_best_diff)
    corr_pct_best_styled = corr_pct_best.style.hide(axis='index').format(precision=2).set_table_styles(
          [{"selector": "", "props": [("border", "1px solid grey"), ('text-align', 'center'), ('max-width', '100%'), ('white-space', 'nowrap')]},
            {"selector": "tbody td", "props": [("border", "1px solid grey"), ('text-align', 'center'), ('max-width', '100%'), ('white-space', 'nowrap')]},
          {"selector": "th", "props": [("border", "1px solid black"), ('text-align', 'center'), ('max-width', '100%'), ('white-space', 'nowrap')]},
          ]).set_properties(**{'max-width': '100%', 'white-space': 'nowrap'}).highlight_max(axis=0, props='font-weight: bold; color: blue', subset=pd.IndexSlice[:, max_cols]).highlight_min(axis=0, props='font-weight: bold; color: blue', subset=pd.IndexSlice[:, min_cols]).apply(lambda x: corr_pct_best_diff.applymap(color_cells), axis=None, subset=pd.IndexSlice[:, corr_pct_best.columns[1:]]).applymap(highlight_baseline, subset=pd.IndexSlice[corr_pct_best[('edge', 'corr', 'pct')] == 'BASELINE', :])
    dfi.export(corr_pct_best_styled, out_dir + '/' + str(feature_count) + '.corr_pct_best.png', table_conversion='selenium')
all_df = pd.concat(all_dfs, axis=0)
all_df = all_df.droplevel(level=1, axis=0)
all_df = all_df.loc[all_df[('edge', 'corr', 'pct')] != 'BASELINE', :]
all_df.index.names = pd.MultiIndex.from_tuples([('', '', '#met')])
all_df = all_df.reset_index()
all_df.to_csv(out_dir +'/all_df.tsv', sep='\t', index=False)
all_df = all_df.drop(columns=[('edge', 'corr', 'pct'), ('edge', 'corr', 'type')])
all_gb = all_df.groupby(by=[('', '', '#met')])
all_max = all_gb.max()
all_min = all_gb.min()
all_best = all_max.multiply(max_mask) + all_min.multiply(min_mask)
cols = np.array(all_best.columns)
max_cols = cols[np.where(better_direction > 0)[0]]
min_cols = cols[np.where(better_direction < 0)[0]]
all_best = all_best.reset_index()
all_best_styled = all_best.style.hide(axis='index').format(precision=2).set_table_styles(
          [{"selector": "", "props": [("border", "1px solid black"), ('text-align', 'center'), ('max-width', '100%'), ('white-space', 'nowrap')]},
            {"selector": "tbody td", "props": [("border", "1px solid black"), ('text-align', 'center'), ('max-width', '100%'), ('white-space', 'nowrap')]},
          {"selector": "th", "props": [("border", "1px solid black"), ('text-align', 'center'), ('max-width', '100%'), ('white-space', 'nowrap')]},
          ]).set_properties(**{'max-width': '100%', 'white-space': 'nowrap'}).highlight_max(axis=0, props='font-weight: bold; color: blue', subset=pd.IndexSlice[:, max_cols]).highlight_min(axis=0, props='font-weight: bold; color: blue', subset=pd.IndexSlice[:, min_cols])
dfi.export(all_best_styled, out_dir + '/feature_count_best.png', table_conversion='selenium')
