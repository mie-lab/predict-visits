import torch
from predict_visits.model.transforms import TransformFF, NoTransform
from predict_visits.model.graph_resnet import VisitPredictionModel
from predict_visits.model.fully_connected import FullyConnectedModel

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
        # TODO: add batch norm etc here
        "model_cfg": {
            "out_dim": 1,
            "layers": [128, 64],
            "final_act": "sigmoid",
        },
    },
}
