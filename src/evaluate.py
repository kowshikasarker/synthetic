import os, sys
from pathlib import Path

def prepare_evaluate_substitution_script():
    base_result_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Result'
    evaluate_script = '/shared/nas/data/m1/ksarker2/Synthetic/Code/evaluate_synthesis.py'
    command_dir = base_result_dir + '/Command'
    Path(command_dir).mkdir(exist_ok=True, parents=True)
    
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
    
    '''datasets = [
        'YACHIDA_CRC_2019',
        'iHMP_IBDMDB_2019',
        'FRANZOSA_IBD_2019',
        'ERAWIJANTARI_GASTRIC_CANCER_2020',
        'MARS_IBS_2020',
        'KOSTIC_INFANTS_DIABETES_2015',
        'JACOBS_IBD_FAMILIES_2016',
        'KIM_ADENOMAS_2020',
        'SINHA_CRC_2016'
    ]'''
    
    datasets = ['SINHA_CRC_2016']
    
    for dataset in datasets:
        print('dataset', dataset)
        filepath = command_dir + '/Evaluate.' + dataset + '.sh'
        fp = open(filepath, 'w')
        train_mic_path = base_result_dir + '/PRADA/' + dataset + '/config-1/preprocess/train_microbiome.tsv'
        train_met_path = base_result_dir + '/PRADA/' + dataset + '/config-1/preprocess/train_metabolome.tsv'
        train_meta_path = base_result_dir + '/PRADA/' + dataset + '/config-1/preprocess/train_metadata.tsv'
        
        test_mic_path = base_result_dir + '/PRADA/' + dataset + '/config-1/preprocess/test_microbiome.tsv'
        test_met_path = base_result_dir + '/PRADA/' + dataset + '/config-1/preprocess/test_metabolome.tsv'
        test_meta_path = base_result_dir + '/PRADA/' + dataset + '/config-1/preprocess/test_metadata.tsv'
        for method in methods:
            print('method', method)
            method_result_dir = base_result_dir + '/' + method + '/' + dataset
            for subdir in method_dirs[method]:
                print('subdir', subdir)
                working_dir = method_result_dir + '/' + subdir + '/output'
                if (method_hparam[method]):
                    best_hparam_path = open(working_dir + '/logs/csv_logs/best_hparam.txt', 'r')
                    best_hparam = best_hparam_path.readline().strip()
                    working_dir = working_dir + '/' + best_hparam
                command = 'python3 ' + evaluate_script + \
                ' --train_mic_path ' + train_mic_path + \
                ' --train_met_path ' + train_met_path + \
                ' --train_meta_path ' + train_meta_path + \
                ' --test_mic_path ' + test_mic_path + \
                ' --test_met_path ' + test_met_path + \
                ' --test_meta_path ' + test_meta_path + \
                ' --syn_mic_path ' + working_dir + '/synthetic_microbiome.tsv' + \
                ' --syn_met_path ' + working_dir + '/synthetic_metabolome.tsv' + \
                ' --syn_meta_path ' + working_dir + '/synthetic_metadata.tsv' + \
                ' --out_dir ' + working_dir + '/evaluate' + \
                ' &'
                fp.write(command + '\n\n')
                fp.flush()
                print(command, end='\n')
        fp.close()
        os.system('chmod +x ' + filepath)

if __name__ == "__main__":
    prepare_evaluate_substitution_script()