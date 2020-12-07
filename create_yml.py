import yaml
import os


if __name__ == "__main__":

    configs = {'model_params': dict(), 'dataset_params': dict(), 'experiment': dict()}

    
    configs['experiment']['train'] = True
    configs['experiment']['link_prediction'] = False
    configs['experiment']['load_model'] = False
    configs['experiment']['load_model_path'] = []

    configs['model_params']['model_name'] = model_name =  'GVAE'    # Choose between [GVAE, GCVAE]
    configs['model_params']['z_dim'] = 400                          # Latent dimensions
    configs['model_params']['h_dim'] = 1024                          # Hidden dimensions
    configs['model_params']['n'] = 1                                # Triples per graph
    configs['model_params']['beta'] = 100                           # betaVAE ratio
    configs['model_params']['epochs'] = 200
    configs['model_params']['lr'] = 3e-7
    configs['model_params']['batch_size_exp2'] = 11                 # 2**(batch_size_exp) fb_max=13 wn_max=12

    configs['dataset_params']['dataset_name'] = dataset = 'wn18rr'  # Choose between [fb15k, wn18rr]



    folder = 'configs/beta/'
    if not os.path.isdir(folder):
        os.makedirs(folder)

        
    for beta in [0.1, 1, 10, 100, 1000]:
        if beta == 0.1:
            configs['experiment']['exp_name'] = 'tune_{}_{}_b{}'.format(model_name, dataset, '01')  # Most important
        else:
            configs['experiment']['exp_name'] = 'tune_{}_{}_b{}'.format(model_name, dataset, beta)  # Most important

        configs['model_params']['beta'] = beta
        configs['model_params']['lr'] = min([(3e-5 / beta), 1e-7])
        yml_name = '{}.yml'.format(configs['experiment']['exp_name'])
        with open(folder + yml_name,'w') as f:
            yaml.dump(configs, f)
