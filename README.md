# Relational GraphVAE

<a href="https://github.com/INDElab/rgvae/graphs/contributors" alt="Contributors">
        <img src="https://img.shields.io/github/contributors/INDElab/rgvae?style=plastic" /></a>
        
<a href="https://github.com/INDElab/rgvae/pulse" alt="Activity">
        <img src="https://img.shields.io/github/commit-activity/m/INDElab/rgvae?style=plastic" /></a>

Implementation of a RGVAE for relational graphs, e.g knowledge graphs.

## Dependencies
[![Python](https://img.shields.io/badge/Python-v3.8-blue?style=plastic)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-v16.0-red?style=plastic)](https://pypi.org/project/torch/)


Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the dependencies.

```bash
pip install -r requirements.txt
```

Now, you can install the RG-VAE module with: 

```bash
pip install -e .
```

## Usage

```bash
python ./experiments/train_eval_GVAE.py
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
