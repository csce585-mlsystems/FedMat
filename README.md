# FedMat
CSCE 585 - Machine Learning Systems Project <br>
**Title**: Federated Learning for Materials Property Prediction <br>

**Team member 1:** <br>
	- **Name**: Sadman Sadeed Omee <br>
	- **Major:** Computer Science (Ph.D. Candidate) <br>
	- **Role:** Machine learning researcher <br>
	- **Email:** somee@email.sc.edu <br>

**Team member 2:** <br>
	- **Name**: Md. Hasibul Amin <br>
	- **Major:** Computer Engineering (Ph.D. Candidate) <br>
	- **Role:** Machine learning researcher <br>
	- **Email:** ma77@email.sc.edu <br>

# Table of Contents
* [Necessary Installations](#installation)
* [Key Features](#key)
* [Project Highlights](#ph)
* [Datasets](#datasets)
* [How to Run](#usage)
* [Project Presentation Video](#video)
* [Acknowledgement](#acknowledgement)

<a name="installation"></a>
## Necessary Installations
Please install the following packages if not already installed. We show how to install them using **pip** only, but you can also use **conda** for the installation purpose. Also you can a virtual environment using conda or pip for this purpose (recommended).

Use the following commands to install the necessary packages:
```bash
git clone https://github.com/csce585-mlsystems/FedMat.git
cd FedMat/FedMat
pip install -r requirements.txt
```
<a name="key"></a>
## Key Features
- **Federated Learning Framework**: Implements FL with support for scalability experiments (varying number of clients) and privacy preservation.
- **Advanced GNN Architectures**: Includes state-of-the-art models like DeeperGATGNN, SchNet, and MPNN, developed for materials property prediction.
- **Benchmark Datasets**: Supports training and evaluation on standard materials datasets, such as those for bandgap, formation energy, and dielectric properties.
- **Out-of-Distribution (OOD) Generalization**: Evaluates the models' ability to generalize to unseen distributions, highlighting challenges in FL for materials science.
- **Scalability Experiments**: Tests the framework's performance with varying client numbers, from small-scale to large-scale federated setups.

<a name="ph"></a>
## Project Highlights
- **Privacy-Aware Training**: FL enables collaboration across multiple institutions without sharing sensitive or proprietary data. Companies/instituitions can share their model without explicityly sahring the name of the special materials they synthesized.
- **Detailed Performance Analysis**: Performance compariosn on in-distribution (ID) and OOD datasets, along with MAE vs. communication round plots.
- **Scalability Insights**: Provides analysis on the trade-offs between client scalability, model performance, and memory usage.

A general pipeline and the federated leaning basic algorithm are shown below:
![](FedMat/figures/pipeline.png)

<a name="datasets"></a>
## Datasets
Materials datasets are large in memory. We provided links from where you can download the datasets and downloading instructions. Additionally, we have provided an already downloaded test data to run to model for a quick test.

To access the datasets, go to the following folder and access the instructions:
```bash
cd FedMat/data/datasets/
```

<a name="usage"></a>
## How to Run
The default configuration for training federated DeeperGATGNN, federated SchNet, and federated MPNN are mentioned in the ```config.yml``` file. First go to the actual code folder using:
```bash
cd FedMat
```
Use the following command to run the federated training of a specific model (DeeperGATGNN shown here):
```bash
python main.py -dataset mat -model deepergatgnn -fedmid avg -part_alpha 0.1 -numClient 4
```

This command will **train the model in a federated way, evaluate the model on test data, generate necessary figures** and store them. Here you can change the dataset and model name to get results for different models on different datasets. Additionally, you can change the number of clients for scalability experiments.

Make sure to unzip the data in FedMat/data/datasets/test_data/ to run it. This command will run for the already downloaded test data. You may download other dataset from the instruction provided in the dataset folders to test the models on them. The command line options for the dataset are mat, band, 2d, alloy, formation, pt, dielectric, gvrh, and perovskites. The command line options for the model are deepergatgnn, schnet, and mpnn.

<a name="vide"></a>
## Project Presentation Video
The project presentation video can be found by clicking [here](https://youtu.be/j_hZIPUTBZ0?si=u3-4A457fwbogc42).

<a name="acknowledgement"></a>
## Acknowledgement
Our code is based on the [FedChem](https://www.sciencedirect.com/science/article/pii/S2666389922001180) algorithm's repository, which has a well-developed federated learning pipeline for molecular data. We adopted this code to make it suitable for the materials datasets.
