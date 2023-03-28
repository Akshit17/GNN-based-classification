# ML4SCI DeepFalcon Tasks

This repository contains evaluation Common Task 2 listed [here](https://docs.google.com/document/d/1bwRaHc0IYIcFOokMcW-mYJv2i24iP1mm08ALTSyQ4EI/edit#)

Common Task 1: https://github.com/Akshit17/Variational-Autoencoder-for-Quark-Gluon-Jet-Event-Images

Common Task 2:  https://github.com/Akshit17/GNN-based-classification

Specific Task n: *add link to specific task*

---
## Quark/Gluon classification



---
## Dataset
[Dataset](https://drive.google.com/file/d/1WO2K-SfU2dntGU4Bb3IYBp9Rh7rtTYEr/view?usp=sharing) given consists of four keys X_jets(image), m0(mass), pt(transverse momentum), y(labels for quark and gluon jet). 
Each image is 125x125 consisting of three channels Track, ECAL and HCAL respectively.

#### Combined channels image sample :-
![Combined channels sample 1](?raw=true)

#### Point cloud visualization :-
![Point cloud for sample 1](?raw=true)

Note :- For visualization purposes Tracks, ECAL, HCAL channels have z = 0,1,2 respectively. However when training 2D surface would be used i.e z=0 for all channels

#### Point cloud to graph representation :-
 * Extracted the non-zero pixels from each channel of the image using a mask.
 * Each image is essentially transformed into a set of nodes, where each node represents a non-zero pixel in the image. 
 * To form the edges between nodes k-nearest neighbor graph was constructed, with k=10. Each node will connected to its k-nearest neighbors.
 * Node features, labels and the adjacency matrix returned by `kneighbors_graph` function from the `sklearn.neighbors` were used to create PyTorch Geometric `Data` objects that could be used as inputs later for the GraphSAGE model.


---
## Model building and training

#### Architecture :-

```
GraphLevelGNN(
  (model): GraphSAGE(
    (conv1): SAGEConv(3, 32, aggr=mean)
    (conv2): SAGEConv(32, 64, aggr=mean)
    (conv3): SAGEConv(64, 128, aggr=mean)
    (lin1): Linear(in_features=128, out_features=64, bias=True)
    (lin2): Linear(in_features=64, out_features=16, bias=True)
    (lin3): Linear(in_features=16, out_features=2, bias=True)
  )
  (loss_module): CrossEntropyLoss()
)
```
Hyperparameters :- `optimizer : Adam`, `learning rate : 1e-3`

For training the model pytorch lighning was used as wrapper 

#### Performance :-

| Model | Test Accuracy | Validation Accuracy | 
| :-------: | :----: | :----: | 
| GraphSAGE (k=10) | 0.7138 | 0.6996 | 
| GraphSAGE (k=4) | 0.7121 | 0.6962 | 
| GraphSAGE (k=2) |  0.7198 | 0.6929 | 



## Discussion






