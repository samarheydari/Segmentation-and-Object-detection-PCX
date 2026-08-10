import math
import os

import click
import numpy as np
import torch
from crp.helper import get_layer_names, load_statistics, load_maximization
from torchvision.transforms import transforms, InterpolationMode
from tqdm import tqdm

from datasets import get_dataset
from datasets.coco2017 import xywhn2xyxy
from models import get_model

from utils.crp import ChannelConcept, FeatureVisualizationSegmentation, FeatureVisualizationLocalization
from utils.crp_configs import ATTRIBUTORS, CANONIZERS, VISUALIZATIONS
from utils.zennit_composites import EpsilonPlus, GuidedBackpropComposite, GradientComposite


@click.command()
@click.option("--model_name", default="yolov6")
@click.option("--dataset_name", default="coco2017")
@click.option("--batch_size", default=10)
@click.option("--layer_name", default="backbone.ERBlock_3.0.rbr_dense.conv")
@click.option("--background_class", default=-1)
@click.option("--class_id", default=0)
def main(model_name, dataset_name, batch_size, layer_name, background_class, class_id):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_dataset, n_classes = get_dataset(dataset_name=dataset_name).values()
    dataset = test_dataset()

    model = get_model(model_name=model_name, classes=n_classes)
    model = model.to(device)
    model.eval()

    attribution = ATTRIBUTORS[model_name](model)
    composite = EpsilonPlus(canonizers=[CANONIZERS[model_name]()])
    composite_guided = GuidedBackpropComposite(canonizers=[CANONIZERS[model_name]()])
    composite_grad = GradientComposite(canonizers=[CANONIZERS[model_name]()])
    cc = ChannelConcept()

    layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])
    layer_map = {layer: cc for layer in layer_names}
    fv = VISUALIZATIONS[model_name](attribution, dataset, layer_map, preprocess_fn=lambda x: x, path=f"{model_name}_{dataset_name}", max_target="max")

    classes = [c for c in np.arange(n_classes) if c != background_class] if class_id is None else [class_id]
    print("Getting most relevant concepts.")
    most_relevant_concepts = get_most_relevant_concepts(model_name, dataset_name, layer_name, classes, 50)

    methods = [
        {"label": "CRP Relevance", "context": []},
        {"label": "LRP", "context": []},
        {"label": "Guided GradCAM", "context": []},
        {"label": "Guided GradCAM Abs", "context": []},
        {"label": "GradCAM", "context": []},
        {"label": "SSGradCAM", "context": []},
        {"label": "SSGradCAM Abs", "context": []},
        {"label": "Activation", "context": []},
    ]

    per_concept = 15

    if class_id is None:
        d_c_sorted, rel_c_sorted, rf_c_sorted = load_maximization(fv.RelMax.PATH, layer_name)
    else:
        d_c_sorted, rel_c_sorted, rf_c_sorted = load_statistics(fv.RelStats.PATH, layer_name, class_id)

    for c_id in tqdm(most_relevant_concepts):

        d_indices = d_c_sorted[:, c_id]
        r_values = rel_c_sorted[:, c_id]


        data_batch, targets_multi = fv.get_data_concurrently(d_indices, preprocessing=True)
        targets_single = []

        for i_t, target in enumerate(targets_multi):
            single_targets = fv.multitarget_to_single(target)
            for st in single_targets:
                targets_single.append(st)

        targets = np.zeros(len(data_batch)).astype(int)
        for t in np.arange(n_classes):
            try:
                target_stats = load_statistics(fv.RelStats.PATH, layer_name, t)
                td_indices = target_stats[0][:, c_id]
                tr_values = target_stats[1][:, c_id]
                cond = [True if (x in td_indices) and (tr_values[list(td_indices).index(x)] == r) else False
                        for x, r in
                        zip(d_indices, r_values)]
                targets[cond] = int(t)
            except FileNotFoundError:
                continue

        data = data_batch[targets != background_class][:per_concept]
        d_indices = d_indices[targets != background_class][:per_concept]
        targets = targets[targets != background_class][:per_concept]

        if isinstance(fv, FeatureVisualizationSegmentation):
            masks = [fv.dataset[ind][1] == t for t, ind in zip(targets, d_indices)]
        elif isinstance(fv, FeatureVisualizationLocalization):
            masks = [get_mask_from_bbxs(fv.dataset[ind][1], t) for t, ind in zip(targets, d_indices)]

        n_samples = len(data)
        if n_samples > batch_size:
            batches = math.ceil(n_samples / batch_size)
        else:
            batches = 1
            batch_size = n_samples

        CRPr_heatmaps = []
        GuidedGradCAM_heatmaps = []
        GuidedGradCAM_abs_heatmaps = []
        GradCAM_heatmaps = []
        SSGradCAM_heatmaps = []
        SSGradCAM_abs_heatmaps = []
        LRP_maps = []
        act_maps = []

        for b in range(batches):
            data_batch = data[b * batch_size: (b + 1) * batch_size]
            targets_batch = targets[b * batch_size: (b + 1) * batch_size]
            resize = transforms.Resize((data.shape[-2], data.shape[-1]), interpolation=InterpolationMode.NEAREST)

            attr = attribution(data_batch, [{layer_name: [c_id], "y": t} for t in targets_batch],
                               composite, record_layer=[layer_name])

            CRPr_heatmaps.append(attr.heatmap.clamp(min=0).detach().cpu())
            LRP_maps.append(attr.relevances[layer_name][:, c_id].clamp(min=0).detach().cpu())
            act_maps.append(attr.activations[layer_name][:, c_id].clamp(min=0).detach().cpu())

            attr_guided = attribution(data_batch, [{layer_name: [c_id], "y": t} for t in targets_batch],
                                        composite_guided, record_layer=[layer_name])
            attr_grad = attribution(data_batch, [{layer_name: [c_id], "y": t} for t in targets_batch],
                                        composite_grad, record_layer=[layer_name])
            GradCAM_heatmaps.append((
                attr_guided.activations[layer_name][:, c_id].clamp(min=0)
                * attr_grad.relevances[layer_name][:, c_id].mean((1, 2))[..., None, None]).clamp(min=0).detach().cpu())

            SSGradCAM_heatmaps.append((
                GradCAM_heatmaps[-1]
                * attr_grad.relevances[layer_name][:, c_id].abs().detach().cpu()))
            SSGradCAM_abs_heatmaps.append((
                act_maps[-1]
                * attr_grad.relevances[layer_name][:, c_id].abs().detach().cpu()))

            GuidedGradCAM_heatmaps.append((
                                    attr_guided.heatmap.clamp(min=0).detach().cpu()
                                    * resize(GradCAM_heatmaps[-1])).clamp(min=0))
            GuidedGradCAM_abs_heatmaps.append((
                                    attr_guided.heatmap.clamp(min=0).detach().cpu()
                                    * resize(act_maps[-1])).clamp(min=0))

            # plot heatmaps in a grid using matplotlib
            # fig, axs = plt.subplots(1, 5, figsize=(20, 4))
            # axs[0].imshow(data_batch[0].permute(1, 2, 0).detach().cpu().numpy())
            # axs[0].set_title("Input")
            # axs[1].imshow(CRPr_heatmaps[-1][0].detach().cpu().numpy())
            # axs[1].set_title("CRP Relevance")
            # axs[2].imshow(LRP_maps[-1][0].detach().cpu().numpy())
            # axs[2].set_title("LRP")
            # axs[3].imshow(GuidedGradCAM_heatmaps[-1][0].detach().cpu().numpy())
            # axs[3].set_title("Guided GradCAM")
            # axs[4].imshow(GradCAM_heatmaps[-1][0].detach().cpu().numpy())
            # axs[4].set_title("GradCAM")
            # plt.show()

            if isinstance(fv, FeatureVisualizationLocalization):
                masks_batch = masks[b * batch_size: (b + 1) * batch_size]
                masks_batch = filter_masks(attribution.model.predict_with_boxes(data_batch), targets_batch, masks_batch)
                masks[b * batch_size: (b + 1) * batch_size] = masks_batch

        CRPr_context = np.mean([(h * (1 - 1 * m)).sum() / (h.sum() + 1e-12)
                                for h, m in zip(torch.cat(CRPr_heatmaps), masks)])
        GGCAM_context = np.mean([(h * (1 - 1 * m)).sum() / (h.sum() + 1e-12)
                                 for h, m in zip(torch.cat(GuidedGradCAM_heatmaps), masks)])
        GGCAM_abs_context = np.mean([(h * (1 - 1 * m)).sum() / (h.sum() + 1e-12)
                                 for h, m in zip(torch.cat(GuidedGradCAM_abs_heatmaps), masks)])

        feature_map_res = (LRP_maps[0].shape[-2], LRP_maps[0].shape[-1])
        resize_mask = transforms.Resize(feature_map_res, interpolation=InterpolationMode.NEAREST)
        masks_ = [resize_mask(m[None])[0] for m in masks]
        LRP_context = np.mean([(h * (1 - 1 * m)).sum() / (h.sum() + 1e-12) for h, m in zip(torch.cat(LRP_maps), masks_)])
        CAM_context = np.mean(
            [(h * (1 - 1 * m)).sum() / (h.sum() + 1e-12) for h, m in zip(torch.cat(GradCAM_heatmaps), masks_)])
        act_context = np.mean([(h * (1 - 1 * m)).sum() / (h.sum() + 1e-12) for h, m in zip(torch.cat(act_maps), masks_)])
        SSCAM_context = np.mean(
            [(h * (1 - 1 * m)).sum() / (h.sum() + 1e-12) for h, m in zip(torch.cat(SSGradCAM_heatmaps), masks_)])
        SSCAM_abs_context = np.mean(
            [(h * (1 - 1 * m)).sum() / (h.sum() + 1e-12) for h, m in zip(torch.cat(SSGradCAM_abs_heatmaps), masks_)])

        methods[0]["context"].append(CRPr_context.item())   # CRP
        methods[1]["context"].append(LRP_context.item())    # LRP
        methods[2]["context"].append(GGCAM_context.item())  # Guided GradCAM
        methods[3]["context"].append(GGCAM_abs_context.item())  # Guided GradCAM Abs
        methods[4]["context"].append(CAM_context.item())    # GradCAM
        methods[5]["context"].append(SSCAM_context.item())  # SSGradCAM
        methods[6]["context"].append(SSCAM_abs_context.item())  # SSGradCAM Abs
        methods[7]["context"].append(act_context.item())    # Activation


    for method in methods:
        method["concepts"] = most_relevant_concepts

    path = f"results/concept_backgroundiness/{dataset_name}/{model_name}"
    os.makedirs(path, exist_ok=True)
    if class_id is not None:
        torch.save(methods, f"{path}/{layer_name}_{class_id}.pth")
    else:
        torch.save(methods, f"{path}/{layer_name}.pth")


def get_most_relevant_concepts(model_name, dataset_name, layer_name, classes, num_concepts):
    top_concepts = []
    for c in classes:
        print(f"Loading class {c} relevances.")
        data = torch.load(f"results/global_class_concepts/{dataset_name}/{model_name}/logits/{layer_name}_class_{c}.pth")
        mean_rel = torch.stack(data["crvs_zplus"]).mean(0)
        num_concepts = min(mean_rel.shape[-1], num_concepts)
        top_concepts.extend(list(torch.topk(mean_rel, num_concepts).indices.detach().cpu().numpy()))
    return np.unique(top_concepts)

def get_mask_from_bbxs(bbxs, target):
    bbxs_class = bbxs[:, 1].long()
    if target not in bbxs_class.numpy():
        print("NNO", bbxs_class.shape)
        return torch.zeros((max(bbxs.shape[0], 1), 640, 640)) == 1
    bbxs = torch.stack([bbxs[i, 2:] for i, t in enumerate(bbxs_class) if t == target])
    bbxs = xywhn2xyxy(bbxs)
    mask = torch.zeros((max(bbxs.shape[0], 1), 640, 640))
    for i, b in enumerate(bbxs):
        mask[i, int(b[1]):int(b[3]), int(b[0]):int(b[2])] = 1
    return mask == 1

def filter_masks(model_output, targets_batch, masks):
    scores, bboxes = model_output
    ms = []
    for i in range(len(scores)):
        s = scores[i]
        b = bboxes[i]
        t = targets_batch[i]
        m = masks[i]
        if m.shape[0] == 1:
            ms.append(m[0])
        else:
            if torch.sum(s.argmax(1) == t) == 0:
                print("no predicted box?")
                ms.append(m[0])
                continue
            b = b[s.argmax(1) == t]
            s = s[s.argmax(1) == t]
            b = b[torch.argmax(s[:, t])]
            predicted_mask = torch.zeros_like(m[0])
            predicted_mask[int(b[1]):int(b[3]), int(b[0]):int(b[2])] = 1
            ious = (m * predicted_mask[None]).sum((1, 2)) / (((m + predicted_mask[None]) >= 1).sum((1, 2)))
            if ious.argmax().item() < 0.3:
                ms.append(predicted_mask)
            else:
                ms.append(m[ious.argmax().item()])
    return ms

if __name__ == "__main__":
    main()
