# Graph-based prior-guided synthetic metabolomic data generation
## Usage
### Preprocessing
```
python3 preprocess.py --feature_path FEATURE_PATH --condition_path CONDITION_PATH --missing_pct MISSING_PCT --imputation {knn} --feature_count FEATURE_COUNT --train_pct TRAIN_PCT --val_pct VAL_PCT
                     [--corr_method {sp,pr,dcov,dcol} [{sp,pr,dcov,dcol} ...]] [--corr_top_pct CORR_TOP_PCT] [--prior_path PRIOR_PATH] [--prior_top_pct PRIOR_TOP_PCT] --out_dir OUT_DIR
```
