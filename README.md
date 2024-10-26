# Graph-based prior-guided synthetic metabolomic data generation
## Usage
### Preprocess
Preprocesses metabolomic data: discards columns with too many missing values, normalizes rows with row sum, imputes remaining missing values, normalizes rows with row sum again, standardizes columns and selects top columns based on least anova p-values
```
python3 preprocess.py --feature_path FEATURE_PATH --condition_path CONDITION_PATH --missing_pct MISSING_PCT --imputation {knn} --feature_count FEATURE_COUNT --train_pct TRAIN_PCT --val_pct VAL_PCT
                     [--corr_method {sp,pr,dcov,dcol} [{sp,pr,dcov,dcol} ...]] [--corr_top_pct CORR_TOP_PCT] [--prior_path PRIOR_PATH] [--prior_top_pct PRIOR_TOP_PCT] --out_dir OUT_DIR
```
#### Arguments
```
  --feature_path FEATURE_PATH
                        path to the features in .tsv format, the first column name should be 'Sample' containing sample identifiers and the rest of the columns should contain metabolomic concentrations
  --condition_path CONDITION_PATH
                        path to the one-hot encoded disease status in .tsv format, the first column name should be 'Sample' containing sample identifiers, and the rest of the columns should each denote one disease group and
                        contain either 0 or 1
  --missing_pct MISSING_PCT
                        max percentage of missing values to keep feature columns, only columns which have less missing percentage in at least one disease group are kept
  --imputation {knn}    method to impute values, currently only knn imputation is supported
  --feature_count FEATURE_COUNT
                        no. of metabolites to keep based on the least anova p-values across disease groups
  --train_pct TRAIN_PCT
                        percentage of sample for training data, used to train the generative model
  --val_pct VAL_PCT     percentage of sample for validation data, used to tune hyperparameters of the generative model
  --corr_method {sp,pr,dcov,dcol} [{sp,pr,dcov,dcol} ...]
                        corr measure(s) to use in constructing correlation graph for samples, needs at least one method
  --corr_top_pct CORR_TOP_PCT
                        top percentage of correlation edges to use, for every metabolite this percentage of top correlated metabolites are conencted with correlation edges
  --prior_path PRIOR_PATH
                        path to prior edges in .tsv format, should contain three columns, the first two contianing metabolites and the third column containing the weight of the prior connection
  --prior_top_pct PRIOR_TOP_PCT
                        top percentage of prior edges to use, for every metabolite this percentage of top prior edges are kept based on higehr weights
  --out_dir OUT_DIR     path to the output dir
```
### Our approach: Graph conditional variational autoencoder (GCVAE)
Trains a graph conditional variational autoencoder with train data, tunes hyperparameters with validation data and generates synthetic data after training completes.
```
python3 train_gcvae.py --train_feature_path TRAIN_FEATURE_PATH --train_condition_path TRAIN_CONDITION_PATH --val_feature_path VAL_FEATURE_PATH --val_condition_path VAL_CONDITION_PATH --model_name {combined_hidden}
                      --syn_sample_count SYN_SAMPLE_COUNT --out_dir OUT_DIR
```
#### Arguments
```
  --train_feature_path TRAIN_FEATURE_PATH
                        train features in .tsv format after preprocessing, the first column name should be 'Sample' containing sample identifiers and the rest of the columns should contain metabolomic concentrations
  --train_condition_path TRAIN_CONDITION_PATH
                        train condiiton in .tsv format after preprocessing, the first column name should be 'Sample' containing sample identifiers, and the rest of the columns should each denote one disease group and contain
                        either 0 or 1
  --val_feature_path VAL_FEATURE_PATH
                        validation features in .tsv format after preprocessing, the first column name should be 'Sample' containing sample identifiers and the rest of the columns should contain metabolomic concentrations
  --val_condition_path VAL_CONDITION_PATH
                        validation condiiton in .tsv format after preprocessing, the first column name should be 'Sample' containing sample identifiers, and the rest of the columns should each denote one disease group and contain
                        either 0 or 1
  --model_name {combined_hidden}
                        name of the model architecture
  --syn_sample_count SYN_SAMPLE_COUNT
                        count of synthetic samples to generate
  --out_dir OUT_DIR     path to output dir
```
