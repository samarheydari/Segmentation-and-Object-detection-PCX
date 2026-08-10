import os
import sys

from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from crp.maximization import Maximization
import torch

from LCRP.utils.crp_configs import ATTRIBUTORS, CANONIZERS, VISUALIZATIONS, COMPOSITES


def run_analysis(model_name, model, dataset, output_dir, device):
    # this code is from L-CRP/experiments/glocal_analysis.py
    # to run it on yolov6s6 like in analysis.py, you need to make following changes
    # 1. Update COMPOSITES with yolov6s6 (or how is it called) and EpsilonGammaFlat
    # 2. Update CANONIZERS with yolov6s6 and YoloV6Canonizer
    # 3. Update ATTRIBUTORS with yolov6s6 and CondAttributionLocalization
    # 4. Update VISUALIZATIONS with yolov6s6 and FeatureVisualizationLocalization
    composite = COMPOSITES[model_name](canonizers=[CANONIZERS[model_name]()])

    model = model.to(device)
    model.eval()
    cc = ChannelConcept()
    layer_names = get_layer_names(model, [torch.nn.Conv2d])
    layer_map = {layer: cc for layer in layer_names}

    attribution = ATTRIBUTORS[model_name](model)

    fv = VISUALIZATIONS[model_name](attribution,
                                    dataset,
                                    layer_map,
                                    preprocess_fn=lambda x: x,
                                    path=output_dir,
                                    max_target="max")

    # increase the number of ref images indices in the crp files from 40(default) to 100 to avoid getting fewer ref images as requested after filtering
    NEW_SAMPLE_SIZE = 100
    fv.RelMax.SAMPLE_SIZE = NEW_SAMPLE_SIZE
    fv.ActMax.SAMPLE_SIZE = NEW_SAMPLE_SIZE
    fv.RelStats.SAMPLE_SIZE = NEW_SAMPLE_SIZE
    fv.ActStats.SAMPLE_SIZE = NEW_SAMPLE_SIZE

    # Here running the analysis on the whole dataset, batch_size is 8, checkpoint is 100
    fv.run(composite, 0, len(dataset), batch_size=8, checkpoint=100)