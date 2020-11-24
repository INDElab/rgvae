import torch
from experiments.link_prediction import link_prediction
import argparse
import matplotlib.pylab as plt

if __name__ == "__main__":
    
    # Arg parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt', nargs=1,
                        help="saved pytorch model",
                        type=argparse.FileType('r'))
    arguments = parser.parse_args()
    pt_path = arguments.pt[0]
    device = 'cpu'
    loaded = torch.load(pt_path, map_location=torch.device(device))

    loss_dict = loaded['loss_log']

    fig, ax = plt.subplots()
    ax.plot(zip(*sorted(loss_dict['val'].items())), 'g', label='Validation loss')
    ax.plot(zip(*sorted(loss_dict['train'].items())), 'b', label='Training loss')
    plt.title(pt_path.split('-')[0])
    plt.xlabel('Epoch')
    plt.ylabel('Elbo')

    plt.show()
