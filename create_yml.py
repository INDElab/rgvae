import yaml
import os


if __name__ == "__main__":

    configs = dict()

    configs['project'] = 'knowledge_generation'           # TODO change this to the final experiment name for wandb
    configs['train'] = False
    configs['link_prediction'] = False
    configs['load_model'] = False
    configs['load_model_path'] = []
    configs['eval_generation'] = True

    configs['model_name'] = model_name =  'GVAE'    # Choose between ['GVAE', 'GCVAE']
    configs['z_dim'] = 100                          # Latent dimensions
    configs['h_dim'] = 512                          # Hidden dimensions 2048, 512, 
    configs['n'] = 1                                # Triples per graph
    configs['beta'] = 100                           # betaVAE ratio
    configs['adj_argmax'] = True
    configs['perm_inv'] = True
    configs['softmax_E'] = True

    configs['epochs'] = 80
    configs['lr'] = 3e-5
    configs['batch_size_exp2'] = 13                 # 2**(batch_size_exp) fb_max=13 wn_max=12
    configs['k'] = 5
    configs['clip_grad'] = True
    configs['final'] = True

    configs['dataset_name'] = dataset = 'fb15k'  # ['fb15k', 'wn18rr']



    folder = 'configs/final_kg/'            # TODO change folder
    if not os.path.isdir(folder):
        os.makedirs(folder)

        
    # for dataset in ['fb15k', 'wn18rr']:
    #     for l in [10, 100, 1000]:
    for model_name in ['GVAE', 'GCVAE', 'GCVAE2']:
        configs['exp_name'] = 'kg{}_{}_p{}'.format(model_name, dataset, '1' if configs['perm_inv'] else '0')  # Most important    '1' if configs['perm_inv'] else '0'
        configs['load_model_path'] = 'data/model/yn{}_{}_p{}_20210105/rgvae_dict_final.pt'.format(model_name, dataset, '1' if configs['perm_inv'] else '0')
        configs['model_name'] = model_name
        yml_name = '{}.yml'.format(configs['exp_name'])
        with open(folder + yml_name,'w') as f:
            yaml.dump(configs, f)
