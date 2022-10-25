# ULN framework (Recurrent VLN-BERT based)

## Prerequisites

### Installation

Install the [Matterport3D Simulator](https://github.com/peteanderson80/Matterport3DSimulator). Notice that this code uses the [old version (v0.1)](https://github.com/peteanderson80/Matterport3DSimulator/tree/v0.1) of the simulator, but you can easily change to the latest version which supports batches of agents and it is much more efficient.

Please follow the instructions in [Recurrent-VLN-BERT](https://github.com/YicongHong/Recurrent-VLN-BERT) to prepare the python environment. 


### Data Preparation

Please follow the instructions in [Recurrent-VLN-BERT](https://github.com/YicongHong/Recurrent-VLN-BERT) to prepare the data and PREVALENT weights. 


### Trained Network Weights

- Classifier: `snap`
    - Download the [trained classifier model]() for the classifier model.

- Agent: `snap`
    - Download the [trained network weights [2.5GB]]() for our agent with Granularity-Specific Subnetworks (GSS).

- Explorer: `snap`
    - Download the [trained explorer model]() for uncertainty-based exploration. 


### Reproduce Testing Results

To replicate the performance reported in our paper, load the trained network weights and run validation:
```bash
bash run/test_agent_final.bash
```

### Training

#### Navigator

To train each module from scratch, simply run:
```bash
bash run/train_{agent, classifier, explorer}.bash
```
The trained model will be saved under `snap/`.


## Todo
- [ ] add checkpoints download link