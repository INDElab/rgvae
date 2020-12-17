import yaml
import os


if __name__ == "__main__":

    configs = dict()

    
    configs['train'] = True
    configs['link_prediction'] = False
    configs['load_model'] = False
    configs['load_model_path'] = []

    configs['model_name'] = model_name =  'GCVAE2'    # Choose between ['GVAE', 'GCVAE']
    configs['z_dim'] = 100                          # Latent dimensions
    configs['h_dim'] = 2048                          # Hidden dimensions
    configs['n'] = 1                                # Triples per graph
    configs['beta'] = 100                           # betaVAE ratio
    configs['adj_argmax'] = True
    configs['perm_inv'] = True
    configs['softmax_E'] = False

    configs['epochs'] = 200
    configs['lr'] = 5e-5
    configs['batch_size_exp2'] = 6                 # 2**(batch_size_exp) fb_max=13 wn_max=12
    configs['k'] = 9

    configs['dataset_name'] = dataset = 'fb15k'  # ['fb15k', 'wn18rr']



    folder = 'configs/final/'            # TODO change folder
    if not os.path.isdir(folder):
        os.makedirs(folder)

        
    for dataset in ['fb15k', 'wn18rr']:
        for beta in [1,100,1000]:
            configs['exp_name'] = 'tune_{}_{}_b{}'.format(model_name, dataset, beta)  # Most important
            configs['dataset_name'] = dataset
            yml_name = '{}.yml'.format(configs['exp_name'])
            with open(folder + yml_name,'w') as f:
                yaml.dump(configs, f)
