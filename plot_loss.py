import torch
from experiments.link_prediction import link_prediction
import argparse
import matplotlib.pylab as plt

if __name__ == "__main__":
    
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
    ax.plot(zip(*sorted(loss_dict['val'].items())), 'g', label='Validation loss')
    ax.plot(zip(*sorted(loss_dict['train'].items())), 'b', label='Training loss')
    plt.title(pt_path.split('/')[-1].strip('.pt'))
    plt.xlabel('Epoch')
    plt.ylabel('Elbo')

    plt.savefig('plots/{}.png'.format(plot_name))
