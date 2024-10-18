import os, sys
from pathlib import Path

def prepare_evaluate_substitution_script():
    base_result_dir = '/shared/nas/data/m1/ksarker2/Synthetic/Manuscript/corr-all'
    evaluate_script = '/shared/nas/data/m1/ksarker2/Synthetic/Code/corr-all/evaluate_synthesis.py'
    command_dir = base_result_dir + '/Command'
    Path(command_dir).mkdir(exist_ok=True, parents=True)
        
    configs = {
        'GCVAE': ['config-' + str(i) for i in range(1, 37)],
        'CVAE': ['config-1', 'config-13', 'config-25'],
        'NOISE': ['config-1', 'config-13', 'config-25'],
        'SMOTE': ['config-1', 'config-13', 'config-25']
    }
    
    datasets = ['ST001386']
        
    for dataset in datasets:
        print('dataset', dataset)
        filepath = command_dir + '/Evaluate.' + dataset + '.sh'
        fp = open(filepath, 'w')
        train_feature_path = base_result_dir + '/GCVAE/' + dataset + '/config-1/preprocess/train_feature.tsv'
        train_condition_path = base_result_dir + '/GCVAE/' + dataset + '/config-1/preprocess/train_condition.tsv'
        
        test_feature_path = base_result_dir + '/GCVAE/' + dataset + '/config-1/preprocess/test_feature.tsv'
        test_condition_path = base_result_dir + '/GCVAE/' + dataset + '/config-1/preprocess/test_condition.tsv'
        
        for method in configs.keys():
            print('method', method)
            method_result_dir = base_result_dir + '/' + method + '/' + dataset
            for config in configs[method]:
                print('config', config)
                config_dir = method_result_dir + '/' + config
                preprocess_dir = config_dir + '/preprocess'
                output_dir = config_dir + '/output'
                best_hparam_path = open(output_dir + '/logs/best_hparam.txt', 'r')
                best_hparam = best_hparam_path.readline().strip()
                best_hparam_dir = output_dir + '/' + best_hparam
                command = 'python3 ' + evaluate_script + \
                ' --train_feature_path ' + preprocess_dir + '/train_feature.tsv' + \
                ' --train_condition_path ' + preprocess_dir + '/train_condition.tsv' + \
                ' --test_feature_path ' + preprocess_dir + '/test_feature.tsv' + \
                ' --test_condition_path ' + preprocess_dir + '/test_condition.tsv' + \
                ' --syn_feature_path ' + best_hparam_dir + '/synthetic_feature.tsv' + \
                ' --syn_condition_path ' + best_hparam_dir + '/synthetic_condition.tsv' + \
                ' --out_dir ' + best_hparam_dir + '/evaluate' + \
                ' &'
                fp.write(command + '\n\n')
                fp.flush()
                print(command, end='\n')
        fp.close()
        os.system('chmod +x ' + filepath)

if __name__ == "__main__":
    prepare_evaluate_substitution_script()