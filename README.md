# Visit frequency prediction

This repository contains the code for our [paper](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.GIScience.2023.84) presented as a short paper at GIScience 2023, titled **Predicting visit frequencies to new places**. In this work, we propose to train neural network models on visit frequency prediction, i.e., predicting the number of future visits to a newly visited location. 

Furthermore, we provide the [appendix to our paper](supplementary_information.pdf) in this repo.

Unfortunately, our analysis is based on proprietary datasets and can therefore not be reproduced without access to the data.

### Installation:

The predict-visits code depends on the `graph_trackintel` package. The package is available on the MIE Lab GitHub and can be installed with pip in editable mode. To do so:
```
git clone https://github.com/mie-lab/graph-trackintel.git
```

In an activated virtual environment, cd into this folder and run
```
pip install -e .
```
This will execute the `setup.py` file and install required dependencies


### Preprocessing

First, the graphs are loaded and preprocessed. The data is then saved as a pickle file. 
One dataset can include several studies. The studies to be included are currently hard-coded in the script. Change the list of studies you want to include in the dataset (variable `studies`) and execute
```
python scripts/graph_preprocessing.py -s train_data
```
This will save a pickle file under `data/train_data.pkl` with the graphs of all studies that were specified.

### Train:

The parameters in the training script are still hardcoded. Change them to the desired model name and the correct dataset names, and run
```
python scripts/train.py
```

## References

Please consider citing our paper if you build up on this code:

Wiedemann, N., Hong, Y., & Raubal, M. (2023). Predicting visit frequencies to new places (Short Paper). In 12th International Conference on Geographic Information Science (GIScience 2023). Schloss-Dagstuhl-Leibniz Zentrum für Informatik.

```bib
@inproceedings{wiedemann2023predicting,
  title={Predicting visit frequencies to new places (Short Paper)},
  author={Wiedemann, Nina and Hong, Ye and Raubal, Martin},
  booktitle={12th International Conference on Geographic Information Science (GIScience 2023)},
  year={2023},
  organization={Schloss-Dagstuhl-Leibniz Zentrum f{\"u}r Informatik}
}
```