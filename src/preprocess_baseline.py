from pathlib import Path
from shutil import copy

base_in_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Manuscript/corr-all/GCVAE'

datasets = ['ST001386']

filenames = [
    'train_feature.tsv',
    'train_condition.tsv',
    'val_feature.tsv',
    'val_condition.tsv',
    'test_feature.tsv',
    'test_condition.tsv',
    'preprocessed_feature.tsv',
    'preprocessed_condition.tsv',
]

configs = ['config-1', 'config-13', 'config-25']
#baselines = ['CVAE', 'NOISE', 'SMOTE']
baselines = ['NOISE', 'SMOTE']

for baseline in baselines:
    base_out_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Manuscript/corr-all/' + baseline
    for dataset_no in range(len(datasets)):
        dataset = datasets[dataset_no]
        print(dataset)
        for config in configs:
            out_dir = base_out_dir + '/' + dataset + '/' + config + '/preprocess'
            Path(out_dir).mkdir(exist_ok=True, parents=True)
            for filename in filenames:
                in_path = base_in_dir + '/' + dataset + '/' + config + '/preprocess/' + filename 
                out_path = out_dir + '/' + filename
                print('in_path', in_path)
                print('out_path', out_path)
                copy(in_path, out_path)
            print()