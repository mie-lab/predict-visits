# predict-visits

## Installation:

Unfortunately, to load the data from the server, a module of another repository is required. The reason is that the data on the server is pickled, and therefore the class of the graph object is needed. Run
```
git clone https://github.com/mie-lab/mobility-graph-representation.git
export PYTHONPATH=PYTHONPATH:$PWD/../mobility-graph-representation
```

This repo can then be installed in editable mode. In an activated virtual environment, cd into this folder and run
```
pip install -e .
```
This will execute the `setup.py` file and install required dependencies