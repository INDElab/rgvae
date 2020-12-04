import torch
from experiments.link_prediction import link_prediction
import argparse
import matplotlib.pylab as plt
import os
import seaborn as sns




if __name__ == "__main__":
    
    sns.set()
    
    # Arg parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt', nargs=1,
                        help="absolut path to pytorch model",
                        type=str)

    parser.add_argument('--n', nargs=1,
                        help="name to save plot",
                        type=str)

    arguments = parser.parse_args()
    pt_path = arguments.pt[0]
    plot_name = arguments.n[0]
    device = 'cpu'
    loaded = torch.load(pt_path, map_location=torch.device(device))

    loss_dict = loaded['loss_log']

    fig, ax = plt.subplots()
    ax.plot(*list(zip(*sorted(loss_dict['val'].items()))), 'g', label='Validation loss')
    ax.plot(*list(zip(*sorted(loss_dict['train'].items()))), 'b', label='Training loss')
    plt.legend(loc='upper right')
    plt.title(plot_name)
    plt.xlabel('Epoch')
    plt.ylabel('Elbo')

    plt.savefig(os.path.dirname(os.path.abspath(__file__)) + '/plots/{}.png'.format(plot_name))
