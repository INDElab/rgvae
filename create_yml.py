import yaml
import os


if __name__ == "__main__":

    configs = dict()

    
    configs['train'] = True
    configs['link_prediction'] = False
    configs['load_model'] = False
    configs['load_model_path'] = []

    configs['model_name'] = model_name =  'GVAE'    # Choose between ['GVAE', 'GCVAE']
    configs['z_dim'] = 100                          # Latent dimensions
    configs['h_dim'] = 2048                          # Hidden dimensions 2048, 512, 
    configs['n'] = 1                                # Triples per graph
    configs['beta'] = 100                           # betaVAE ratio
    configs['adj_argmax'] = True
    configs['perm_inv'] = False
    configs['softmax_E'] = True

    configs['epochs'] = 200
    configs['lr'] = 3e-5
    configs['batch_size_exp2'] = 6                 # 2**(batch_size_exp) fb_max=13 wn_max=12
    configs['k'] = 5
    configs['clip_grad'] = False

    configs['dataset_name'] = dataset = 'fb15k'  # ['fb15k', 'wn18rr']



    folder = 'configs/final/'            # TODO change folder
    if not os.path.isdir(folder):
        os.makedirs(folder)

        
    for dataset in ['fb15k', 'wn18rr']:
        configs['exp_name'] = '{}_{}_p{}'.format(model_name, dataset, '1' if configs['perm_inv'] else '0')  # Most important
        configs['dataset_name'] = dataset
        yml_name = '{}.yml'.format(configs['exp_name'])
        with open(folder + yml_name,'w') as f:
            yaml.dump(configs, f)
