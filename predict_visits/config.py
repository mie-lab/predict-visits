import torch
from predict_visits.model.transforms import TransformFF, NoTransform
from predict_visits.model.graph_resnet import VisitPredictionModel
from predict_visits.model.fully_connected import FullyConnectedModel
from predict_visits.model.transformer_model import TransformerModel

model_dict = {
    "gcn": {
        "model_class": VisitPredictionModel,
        "inp_transform": NoTransform,
        "model_cfg": {
            "out_dim": 1,
            "ff_layers": [64, 32],
            "graph_enc_dim": 64,
            "graph_k": 4,
        },
    },
    "ff": {
        "model_class": FullyConnectedModel,
        "inp_transform": TransformFF,
        "model_cfg": {
            "out_dim": 1,
            "layers": [128, 64],
            "final_act": "sigmoid",
        },
    },
    "transformer": {
        "model_class": TransformerModel,
        "inp_transform": TransformFF,
        "model_cfg": {
            "out_dim": 1,
            "flatten": False # this is for the input transform
            # TODO: add further details for the model, like number of layers
        },
    },
}
