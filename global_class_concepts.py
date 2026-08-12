import os
import os
import random
import sys

# Allow running this file directly from IDE "Run" button.
# When executed as a script, Python's import root is this file's folder
# (`.../experiments`), so sibling top-level packages like `datasets` are not
# visible unless we add project root (`.../code`) to sys.path.
project_root = os.path.abspath("/home/heydari/paper/Segmentation-and-Object-detection-PCX/")

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import click
import numpy as np
import torch
from crp.helper import get_layer_names

from tqdm import tqdm

from datasets import get_dataset
from LCRP.models import get_model

from LCRP.utils.crp import ChannelConcept
from LCRP.utils.crp_configs import ATTRIBUTORS, CANONIZERS, VISUALIZATIONS
from LCRP.utils.zennit_composites import EpsilonPlusFlat, EpsilonGammaFlat, EpsilonFlat, GradientComposite


@click.command()
@click.option("--model_name", default="pidnet")
@click.option("--dataset_name", default="flood")
@click.option("--class_id", default=1)
@click.option("--batch_size", default=1, show_default=True,
              help="Attribution batch size. PIDNet at 720x1280 typically requires 1 on a 24 GB GPU.")
@click.option("--rel_init", default="logits", show_default=True, help="[ones, prob, logits]")
@click.option(
    "--checkpoint_path",
    default=os.path.join(project_root, "LCRP/models/flood_model.pt"),
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    show_default=True,
    help="PIDNet checkpoint used to generate the concept vectors.",
)
def main(model_name, dataset_name, class_id, batch_size, rel_init, checkpoint_path):
    model_aliases = {
        "YOLOv6s": "yolov6s6",
        "yolov6s": "yolov6s6",
    }
    model_name = model_aliases.get(model_name, model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_dataset, n_classes = get_dataset(dataset_name=dataset_name).values()
    dataset_kwargs = {}
    if dataset_name == "flood" and model_name == "pidnet":
        dataset_kwargs["resize_hw"] = (384, 512)
        dataset_kwargs["mask_downsample"] = 8
    dataset = test_dataset(**dataset_kwargs)

    model_kwargs = {"classes": n_classes}
    if model_name == "pidnet":
        checkpoint_path = os.path.abspath(checkpoint_path)
        print(f"Loading PIDNet checkpoint: {checkpoint_path}")
        model_kwargs["ckpt_path"] = checkpoint_path
    model = get_model(model_name=model_name, **model_kwargs)
    model = model.to(device)
    model.eval()

    attribution = ATTRIBUTORS[model_name](model)
    setattr(attribution, "rel_init", rel_init)
    lrp_zp_composite = EpsilonPlusFlat(canonizers=[CANONIZERS[model_name]()])
    lrp_gamma_composite = EpsilonGammaFlat(canonizers=[CANONIZERS[model_name]()])
    lrp_eps_composite = EpsilonFlat(canonizers=[CANONIZERS[model_name]()])
    grad_composite = GradientComposite(canonizers=[CANONIZERS[model_name]()])
    cc = ChannelConcept()
    condition = [{"y": class_id}]
    layer_names = get_layer_names(model, [torch.nn.Conv2d])
    if model_name == "pidnet" and dataset_name == "flood":
        # Record only the layers consumed by the paper perturbation figure.
        Post_merge_layers = [
            "base_model.dfm.conv_p.0",
            "base_model.dfm.conv_i.0",
            "base_model.final_layer.conv1",
            "base_model.final_layer.conv2",
        ]
        missing_layers = [layer for layer in Post_merge_layers if layer not in layer_names]
        if missing_layers:
            raise RuntimeError(f"PIDNet paper layers not found: {missing_layers}")
        layer_names = Post_merge_layers
        print("Recording only PIDNet paper layers:", layer_names)

    layer_map = {layer: cc for layer in layer_names}
    fv = VISUALIZATIONS[model_name](attribution,
                                    dataset,
                                    layer_map,
                                    preprocess_fn=lambda x: x,
                                    path=f"{model_name}_{dataset_name}",
                                    max_target="max")

    samples = np.array([i for i in range(len(dataset)) if class_id in fv.multitarget_to_single(fv.get_data_sample(i)[1])])
    n_samples = len(samples)
    n_batches = int(np.ceil(n_samples / batch_size)) if n_samples > 0 else 0

    crvs_zplus = dict(zip(layer_names, [[] for _ in layer_names]))
    crvs_gamma = dict(zip(layer_names, [[] for _ in layer_names]))
    crvs_eps = dict(zip(layer_names, [[] for _ in layer_names]))
    cavs_max = dict(zip(layer_names, [[] for _ in layer_names]))
    cavs_mean = dict(zip(layer_names, [[] for _ in layer_names]))
    crvs_grad = dict(zip(layer_names, [[] for _ in layer_names]))
    crvs_gradcam = dict(zip(layer_names, [[] for _ in layer_names]))
    smpls = []

    for i in tqdm(range(n_batches)):
        samples_batch = samples[i * batch_size:(i + 1) * batch_size]
        data = torch.stack([dataset[j][0] for j in samples_batch], dim=0).to(device).requires_grad_()
        setattr(attribution, "rel_init", rel_init + "_zplus")
        attr_zplus = attribution(data, condition, lrp_zp_composite, record_layer=layer_names, init_rel=1)
        setattr(attribution, "rel_init", rel_init)
        attr_gamma = attribution(data, condition, lrp_gamma_composite, record_layer=layer_names, init_rel=1)
        attr_eps = attribution(data, condition, lrp_eps_composite, record_layer=layer_names, init_rel=1)
        setattr(attribution, "rel_init", rel_init + "_grad")
        attr_grad = attribution(data, condition, grad_composite, record_layer=layer_names, init_rel=1)
        setattr(attribution, "rel_init", rel_init)

        # Some Conv2d layer names can be present in get_layer_names() but missing
        # in returned relevance/activation maps (e.g. detect helper layers).
        active_layers = [
            l for l in layer_names
            if l in attr_zplus.relevances
            and l in attr_gamma.relevances
            and l in attr_eps.relevances
            and l in attr_grad.relevances
            and l in attr_zplus.activations
        ]
        if not active_layers:
            continue

        non_zero = (attr_zplus.heatmap.sum((1, 2)).abs() > 0).detach().cpu().numpy()
        samples_nz = samples_batch[non_zero]
        if samples_nz.size:
            # Store portable Python integers rather than NumPy scalar objects.
            smpls += [int(s) for s in samples_nz]
            rels_zplus = [cc.attribute(attr_zplus.relevances[layer][non_zero], abs_norm=True) for layer in active_layers]
            rels_gamma = [cc.attribute(attr_gamma.relevances[layer][non_zero], abs_norm=True) for layer in active_layers]
            rels_eps = [cc.attribute(attr_eps.relevances[layer][non_zero], abs_norm=True) for layer in active_layers]
            acts_max = [attr_zplus.activations[layer][non_zero].flatten(start_dim=2).max(2)[0] for layer in active_layers]
            acts_mean = [attr_zplus.activations[layer][non_zero].mean((2, 3)) for layer in active_layers]
            gradient = [cc.attribute(attr_grad.relevances[layer][non_zero]) for layer in active_layers]
            gradcam = [cc.attribute(attr_zplus.activations[layer][non_zero].clamp(min=0)
                                    * attr_grad.relevances[layer][non_zero].mean((2, 3))[..., None, None])
                       for layer in active_layers]
            for l, r_zplus, r_gamma, r_eps, amax, amean, grad, gradc in zip(active_layers,
                                                  rels_zplus, rels_gamma, rels_eps,  acts_max, acts_mean, gradient, gradcam):
                crvs_zplus[l] += r_zplus.detach().cpu()
                crvs_gamma[l] += r_gamma.detach().cpu()
                crvs_eps[l] += r_eps.detach().cpu()
                cavs_max[l] += amax.detach().cpu()
                cavs_mean[l] += amean.detach().cpu()
                crvs_grad[l] += grad.detach().cpu()
                crvs_gradcam[l] += gradc.detach().cpu()

        # Attribution results retain large autograd graphs and recorded feature
        # maps. Release them before processing the next 720x1280 batch.
        del attr_zplus, attr_gamma, attr_eps, attr_grad, data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    path = os.path.join(project_root, "results/global_class_concepts", dataset_name, model_name, rel_init)
    os.makedirs(path, exist_ok=True)
    torch.save({"samples": smpls,
                "crvs_zplus": crvs_zplus,
                "crvs_gamma": crvs_gamma,
                "crvs_eps": crvs_eps,
                "cavs_max": cavs_max,
                "cavs_mean": cavs_mean,
                "crvs_grad": crvs_grad,
                "crvs_gradcam": crvs_gradcam},
               f"{path}/class_{class_id}.pth")

    for layer in layer_names:
        torch.save({"samples": smpls,
                    # "output": output,
                    "crvs_zplus": crvs_zplus[layer],
                    "crvs_gamma": crvs_gamma[layer],
                    "crvs_eps": crvs_eps[layer],
                    "cavs_max": cavs_max[layer],
                    "cavs_mean": cavs_mean[layer],
                    "crvs_grad": crvs_grad[layer],
                   "crvs_gradcam": crvs_gradcam[layer]},
                   f"{path}/{layer}_class_{class_id}.pth")


if __name__ == "__main__":
    main()
