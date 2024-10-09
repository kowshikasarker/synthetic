from pathlib import Path
from shutil import copy

base_in_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Result/PRADA'
#base_out_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Result/CVAE'
base_out_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Result/SMOTE'
#base_out_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Result/NOISE'

datasets = [
    'YACHIDA_CRC_2019',
    'iHMP_IBDMDB_2019',
    'FRANZOSA_IBD_2019',
    'ERAWIJANTARI_GASTRIC_CANCER_2020',
    'MARS_IBS_2020',
    'KOSTIC_INFANTS_DIABETES_2015',
    'JACOBS_IBD_FAMILIES_2016',
    'KIM_ADENOMAS_2020',
    'SINHA_CRC_2016'
]

filenames = [
    'train_microbiome.tsv',
    'train_metabolome.tsv',
    'train_metadata.tsv',
    'val_microbiome.tsv',
    'val_metabolome.tsv',
    'val_metadata.tsv',
    'test_microbiome.tsv',
    'test_metabolome.tsv',
    'test_metadata.tsv',
    'preprocessed_microbiome.tsv',
    'preprocessed_metabolome.tsv',
    'preprocessed_metadata.tsv',
]

for dataset_no in range(len(datasets)):
    dataset = datasets[dataset_no]
    print(dataset)
    out_dir = base_out_dir + '/' + dataset + '/preprocess'
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    for filename in filenames:
        in_path = base_in_dir + '/' + dataset + '/config-1/preprocess/' + filename 
        out_path = out_dir + '/' + filename
        print('in_path', in_path)
        print('out_path', out_path)
        copy(in_path, out_path)
    print()