# Agent for RoomEnv-v2

[![DOI](https://zenodo.org/badge/777194227.svg)](https://zenodo.org/doi/10.5281/zenodo.10876430)
[![DOI](https://img.shields.io/badge/Paper-PDF-red.svg)]()

This repo is to train an agent with a GNN that interacts with the [RoomEnv-v2](https://github.com/humemai/room-env).

## Prerequisites

1. A unix or unix-like x86 machine
1. python 3.10 or higher.
1. Running in a virtual environment (e.g., conda, virtualenv, etc.) is highly
   recommended so that you don't mess up with the system python.
1. Install the requirements by running `pip install -r requirements.txt`

## Training

- [Jupyter Notebook for training](train-dqn.ipynb)
- The training results are saved at [`./trained-results`](./trained-results/).

## HumemAI Agent

| A visualization of the two phases involved in training. |
| :-----------------------------------------------------: |
|              ![](./figures/gnn-steps.png)               |

## pdoc documentation

Click on [this link](https://humemai.github.io/agent-room-env-v2-lstm) to see the HTML
rendered docstrings

## Contributing

Contributions are what make the open source community such an amazing place to be learn,
inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
1. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
1. Run `make test && make style && make quality` in the root repo directory, to ensure
   code quality.
1. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
1. Push to the Branch (`git push origin feature/AmazingFeature`)
1. Open a Pull Request

## [Cite our paper]()

```bibtex

```

## Authors

- [Taewoon Kim](https://taewoon.kim/)
- [Michael Cochez](https://www.cochez.nl/)
- [Vincent Francois-Lavet](http://vincent.francois-l.be/)
