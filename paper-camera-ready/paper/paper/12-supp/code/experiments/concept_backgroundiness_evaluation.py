import os

import click
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from datasets import get_dataset
from experiments.concept_backgroundiness import get_mask_from_bbxs
from models import get_model

from utils.crp import ChannelConcept
from utils.crp_configs import ATTRIBUTORS, CANONIZERS
from utils.zennit_composites import EpsilonPlusFlat


@click.command()
@click.option("--model_name", default="yolov6")
@click.option("--dataset_name", default="coco2017")
@click.option("--layer_name", default="backbone.ERBlock_3.0.rbr_dense.conv")
@click.option("--background_class", default=-1)
@click.option("--class_id", default=0)
@click.option("--n_samples", default=60)
def main(model_name, dataset_name, layer_name, background_class, class_id, n_samples):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_dataset, n_classes = get_dataset(dataset_name=dataset_name).values()
    dataset = test_dataset()

    model = get_model(model_name=model_name, classes=n_classes)
    model = model.to(device)
    model.eval()

    attribution = ATTRIBUTORS[model_name](model)
    composite = EpsilonPlusFlat(canonizers=[CANONIZERS[model_name]()])
    setattr(attribution, "rel_init", "logits_zplus")
    cc = ChannelConcept()

    classes = [c for c in np.arange(n_classes) if c != background_class] if class_id is None else [class_id]

    if class_id is not None:
        methods = torch.load(f"results/concept_backgroundiness/{dataset_name}/{model_name}/{layer_name}_{class_id}.pth")
    else:
        methods = torch.load(f"results/concept_backgroundiness/{dataset_name}/{model_name}/{layer_name}.pth")

    most_relevant_concepts = methods[0]["concepts"]
    data = torch.load(f"results/global_class_concepts/{dataset_name}/{model_name}/logits/{layer_name}_class_{class_id}.pth")
    samples = data["samples"]
    activations = []
    relevances = []
    for i, sample in enumerate(tqdm(samples)):
        data, target = dataset[sample]
        if len(relevances) > n_samples:
            break
        if "yolo" in model_name:
            bbxs = target
            target = target[:, 1].long()

        if class_id not in target.unique():
            continue
        for c in target.unique():
            if c not in classes:
                continue
            if "yolo" in model_name:
                masks = get_mask_from_bbxs(bbxs, c.item())
                mask = masks.sum(0) > 0
            else:
                mask = target == c
            if mask.sum() < 500:
                continue
            alphas = np.linspace(0, 1, 3)[1:]
            bgs = [3 * (torch.rand(mask.shape)[None] - 0.5),
                   3 * (torch.rand(data.shape) - 0.5),
                   torch.ones_like(mask)[None] * 0,
                   4 * (torch.ones_like(mask)[None] * torch.rand((3))[:, None, None] - 0.5)]
            ds = [data.clone()]
            for bg in bgs:
                data_masked = data.clone() * mask + bg * (1 - 1 * mask)
                ds += [data.clone() * (1-a) + a * data_masked for a in alphas]

            data = torch.stack(ds).to(device)

            attr = attribution(data.requires_grad_(), [{"y": c}],
                               composite, record_layer=[layer_name])
            relevances.append(cc.attribute(attr.relevances[layer_name], abs_norm=True).detach().cpu())
            activations.append(attr.activations[layer_name].clamp(min=0).flatten(start_dim=2).sum(2).detach().cpu())


    path = f"results/concept_backgroundiness/{dataset_name}/{model_name}"
    torch.save({
        "relevances": torch.stack(relevances),
        "activations": torch.stack(activations),
        "concepts": most_relevant_concepts,
        "backgroundiness": methods
    },
        f"{path}/evaluated_{layer_name}_class_{class_id}.pth")


if __name__ == "__main__":
    main()
