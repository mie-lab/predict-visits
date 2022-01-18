# predict-visits

## Installation:

The predict-visits code depends on the `graph_trackintel` package. The package is available on the MIE Lab GitHub and can be installed with pip in editable mode. To do so:
```
git clone https://github.com/mie-lab/graph-trackintel.git
```

In an activated virtual environment, cd into this folder and run
```
pip install -e .
```
This will execute the `setup.py` file and install required dependencies


## Preprocessing

First, the graphs are loaded and preprocessed. The data is then saved as a pickle file. 
One dataset can include several studies. The studies to be included are currently hard-coded in the script. Change the list of studies you want to include in the dataset (variable `studies`) and execute
```
python scripts/graph_preprocessing.py -s train_data
```
This will save a pickle file under `data/train_data.pkl` with the graphs of all studies that were specified.

## Train:

The parameters in the training script are still hardcoded. Change them to the desired model name and the correct dataset names, and run
```
python scripts/train.py
```
