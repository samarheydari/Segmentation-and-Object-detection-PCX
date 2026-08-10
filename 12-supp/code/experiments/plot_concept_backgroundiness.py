import click
import matplotlib
import numpy as np

import torch
from crp.helper import get_layer_names
import matplotlib.pyplot as plt
from scipy import stats

from datasets import get_dataset

from models import get_model

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

@click.command()
@click.option("--model_name", default="yolov6")
@click.option("--dataset_name", default="coco2017")
@click.option("--background_class", default=-1)
@click.option("--num_classes", default=None, type=int)
@click.option("--labels", default="CRP Relevance,LRP,Guided GradCAM,GradCAM,SSGradCAM,Activation", type=str)
def main(model_name, dataset_name, background_class, num_classes, labels):
    path = f"results/concept_backgroundiness/{dataset_name}/{model_name}"

    _, _, n_classes = get_dataset(dataset_name=dataset_name).values()
    if num_classes is not None:
        n_classes = num_classes

    model = get_model(model_name=model_name, classes=n_classes)
    model.eval()
    layer_names = get_layer_names(model, [torch.nn.Conv2d])

    labels = labels.split(",")

    scores = {}
    MSE = {}
    scores_sensitivity = {}
    scores_context = {}
    layers = []

    if model_name == "unet":
        layer_names = ["encoder.features.0", "encoder.features.5", "encoder.features.10"]
    elif model_name == "yolov6":
        layer_names = [
                       "backbone.ERBlock_3.0.rbr_dense.conv",
                       "backbone.ERBlock_4.0.rbr_dense.conv",
                       "backbone.ERBlock_5.0.rbr_dense.conv"
        ]
    elif model_name == "yolov5":
        layer_names = ["model.6.cv3.conv", "model.8.cv3.conv", "model.10.conv"]
    elif model_name == "deeplabv3plus":
        layer_names = ["backbone.layer4.2.conv2", "backbone.layer4.0.conv3", "backbone.layer3.0.conv1"]

    for layer_name in layer_names[:]:
        classes = [c for c in np.arange(n_classes) if c != background_class]
        relevances = []
        concepts = []
        backgroundiness = []
        for class_id in [c for c in classes]:
            try:
                data = torch.load(f"{path}/evaluated_{layer_name}_class_{class_id}.pth")
                data_ = torch.load(f"results/concept_backgroundiness/{dataset_name}/{model_name}/{layer_name}_{class_id}.pth")
            except:
                classes.remove(class_id)
                print(class_id, "not found")
                continue
            concepts_ = data_[0]["concepts"]
            label_not_in_data = False
            for label in labels:
                if label not in [k["label"] for k in data_]:
                    label_not_in_data = True
                    print(class_id, f"{label} not in data.", layer_name)
                    continue
            if label_not_in_data:
                classes.remove(class_id)
                continue
            if np.sum(data["concepts"] == concepts_) != len(concepts_):
                classes.remove(class_id)
                print(class_id, "NOT SYNCED")
                continue
            if data["relevances"].shape[0] < 20:
                classes.remove(class_id)
                print(class_id, "NOT enough data")
                continue
            relevances.append(data["relevances"])
            concepts.append(data["concepts"])
            backgroundiness.append([d for d in data_ if d["label"] in labels])
        if not backgroundiness:
            continue
        layers.append(layer_name)
        data = {k["label"]: [] for k in backgroundiness[0]}
        datab = {k["label"]: [] for k in backgroundiness[0]}
        for i, bs in enumerate(backgroundiness):
            r = relevances[i]
            c = concepts[i]
            w = r[:, 0, c].clamp(min=0)
            w = w / w.sum(0)[None]
            max_r = torch.where(r.abs() > r[:, 0:1].abs(), r.abs(), r[:, 0:1].abs())
            dr = (r[:, 0:1, c] - r[:, 1:, c]) / (max_r[:, 1:, c] + 1e-12)
            dr = dr.clamp(min=-1, max=1).abs().mean(1)
            Xr = (dr * w).sum(0)
            for b in bs[:]:
                data[b["label"]].append(Xr.numpy())
                datab[b["label"]].append(b["context"])

        plt.figure(dpi=300, figsize=(3, 2.6))

        plt.plot([0, 1], [0, 1], ':', color="gray", linewidth=2, alpha=0.5)
        cmaps = ["Blues", "Greens", "Oranges", "Greys", "Purples", "Greys", "Greys", "Greys"]
        for j, d in enumerate(data):
            if "CRP A" in d:
                continue
            x = np.array(data[d]).flatten()
            y = np.array(datab[d]).flatten()

            kernel = stats.gaussian_kde([x, y], bw_method=0.4)
            X, Y = np.mgrid[-1:2:200j, -1:2:200j]
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = np.reshape(kernel(positions).T, X.shape)
            z0 = 0
            z1 = Z.max()
            z = (z0 + z1) / 2
            goal = 0.5
            for i in range(20):
                frac = Z[Z > z].sum() / Z.sum()
                if frac > goal:
                    z0 = z + 0
                    z = (z + z1) / 2
                else:
                    z1 = z + 0
                    z = (z + z0) / 2
            z05 = z + 0
            z0 = 0
            z1 = Z.max()
            z = (z0 + z1) / 2
            goal = 0.8
            for i in range(20):
                frac = Z[Z > z].sum() / Z.sum()
                if frac > goal:
                    z0 = z + 0
                    z = (z + z1) / 2
                else:
                    z1 = z + 0
                    z = (z + z0) / 2
            z08 = z + 0
            plt.contour(X, Y, Z, levels=[z08, z05], cmap=cmaps[j], alpha=1, linewidths=2, vmin=0, vmax=z05*1.4)
            correlation = np.corrcoef(x, y)[0, 1]
            plt.plot([], [], label=f"{labels[j].replace('CRP Relevance', 'L-CRP').replace('SSGradCAM', 'SS-GradCAM')}",
                     color=matplotlib.colormaps[cmaps[j]](170))

            if d not in scores:
                MSE[d] = [np.mean((x-y)**2)]
                scores[d] = [correlation]
                scores_sensitivity[d] = [x]
                scores_context[d] = [y]
            else:
                MSE[d].append(np.mean((x - y) ** 2))
                scores[d].append(correlation)
                scores_sensitivity[d].append(x)
                scores_context[d].append(y)
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        plt.xlabel("background sensitivity")
        plt.ylabel("context score")
        plt.legend(fontsize=7, loc="lower right")
        plt.tight_layout()
        plt.savefig(f"{path}/plots/context_evaluation_{model_name}_{dataset_name}_{layer_name}_{'_'.join(labels)}.pdf", dpi=300, transparent=True)
        plt.show()

    print("MSE")
    print(MSE)
    for m in MSE:
        s = np.sqrt(np.array(MSE[m])).mean()
        print(m, s)
    print("CORRELATION")
    print(scores)
    for m in scores:
        s = (np.array(scores[m])).mean()
        print(m, s)

### VALUES FROM THE PAPER
# UNET = {'CRP Relevance': [0.05176440303949518, 0.03521040158154467, 0.03156545965869124], 'LRP': [0.052971147100200476, 0.03712349823104667, 0.045074278661918626], 'Guided GradCAM': [0.08061577548035435, 0.0615524526009376, 0.08020021445825502], 'GradCAM': [0.07870107460923352, 0.07833981523745247, 0.07402511616736834], 'SSGradCAM': [0.07083260959921248, 0.05631198759427123, 0.0743713917093751], 'Activation': [0.28083843058391755, 0.24582839969076659, 0.16065709972080947]}
# DEEPLAB = {'CRP Relevance': [0.013994018159422791, 0.025006483904425485, 0.06939290543145153], 'LRP': [0.016598914987635167, 0.032711174280374955, 0.0748816980854307], 'Guided GradCAM': [0.05108592387600268, 0.039206706631565745, 0.023716195783307255], 'GradCAM': [0.058916831222439875, 0.06788095247748491, 0.04783228256341886], 'SSGradCAM': [0.05244064119806518, 0.040853102054040365, 0.02772411330072359], 'Activation': [0.052474614712984646, 0.11921768809146323, 0.239920952285556]}
# YOLOv6 = {'CRP Relevance': [0.029766202947452264, 0.02866799102971604, 0.02525644204808738], 'LRP': [0.02717248448434474, 0.026664829185823368, 0.031931113849614313], 'Guided GradCAM': [0.05890688296154394, 0.06474665206808862, 0.07529994622138021], 'GradCAM': [0.05688684884749784, 0.05754067156502557, 0.045542342283827995], 'SSGradCAM': [0.05193924322141832, 0.05279688336135673, 0.06686602517408669], 'Activation': [0.19811658916846228, 0.20345140879382073, 0.11923832592469194]}# DEEPLAB = {'CRP Relevance': [0.013994018159422791, 0.02500357651897262, 0.06939290543145153], 'LRP': [0.016598914987635167, 0.03270809820519769, 0.0748816980854307], 'Guided GradCAM': [0.05108592387600268, 0.03920571852503578, 0.023716195783307255], 'Guided GradCAM Abs': [0.023060795548827007, 0.044821832953566156, 0.10207273912990515], 'GradCAM': [0.058916831222439875, 0.06787760328197877, 0.04783228256341886], 'SSGradCAM': [0.05244064119806518, 0.040852960969084195, 0.02772411330072359], 'SSGradCAM Abs': [0.025855550772715463, 0.054471415079477864, 0.1328697395657989], 'Activation': [0.052474614712984646, 0.11920946890985705, 0.239920952285556]}
# YOLOv5 = {'CRP Relevance': [0.020515245071324024, 0.031005894270548028, 0.022852148331988887], 'LRP': [0.030216201546139593, 0.030567902965446786, 0.09495701762105178], 'Guided GradCAM': [0.05756882938879573, 0.05293837594446657, 0.08807674911922093], 'GradCAM': [0.06306023447529552, 0.05916558944624149, 0.08009302594748485], 'SSGradCAM': [0.06158120974096802, 0.0558284155257171, 0.14577527206890994], 'Activation': [0.21233562212046506, 0.15076196339801629, 0.07909097062436954]}
# scores = [YOLOv6, YOLOv5, UNET, DEEPLAB]
# for k in YOLOv5:
#     print(k)
#     print(np.mean(np.sqrt([s[k] for s in scores])))
#
# YOLOv5 = {'CRP Relevance': [0.6863563520739825, 0.8258723735230498, 0.828751098736014], 'LRP': [0.5485213158751444, 0.8052374845999426, 0.7176541036899955], 'Guided GradCAM': [0.37611567865824147, 0.5552844782644025, 0.4973820794960415], 'GradCAM': [0.24705495639190111, 0.3316263753421742, 0.33380802949243676], 'SSGradCAM': [0.36945128703472924, 0.5648111593236638, 0.5847976176454048], 'Activation': [0.6390005954968153, 0.7802141097748971, 0.7443805466260836]}
# UNET = {'CRP Relevance': [0.46001976431879715, 0.6031469527516373, 0.6958226577343473], 'LRP': [0.4587145079828479, 0.600443410621417, 0.6584797192016454], 'Guided GradCAM': [0.11025614935144615, 0.35310534322115056, 0.4633599522412759], 'GradCAM': [0.002697156017370452, 0.10687575771374783, 0.2228959298865435], 'SSGradCAM': [0.10667610359608015, 0.29802610190475154, 0.4097793527616098], 'Activation': [0.40970678743681205, 0.5518787991248675, 0.644993851726199]}
# YOLOv6 = {'CRP Relevance': [0.6539133831628918, 0.6835516671271987, 0.8045090953642571], 'LRP': [0.6592594189755407, 0.6886320106602078, 0.8034006961772918], 'Guided GradCAM': [0.5403486961331949, 0.5468707682850182, 0.651899040735896], 'GradCAM': [0.32027245203534727, 0.32125510327471446, 0.435884271047907], 'SSGradCAM': [0.4770785781002401, 0.5151992011379894, 0.6423404220596044], 'Activation': [0.6821041853838256, 0.6819099051201799, 0.7550496877902265]}# UNET = {'CRP Relevance': [0.05176440303949518, 0.03521040158154467, 0.03156545965869124], 'LRP': [0.052971147100200476, 0.03712349823104667, 0.045074278661918626], 'Guided GradCAM': [0.08061577548035435, 0.0615524526009376, 0.08020021445825502], 'Guided GradCAM Abs': [0.08950914214658903, 0.05443665026967033, 0.049059425915577624], 'GradCAM': [0.07870107460923352, 0.07833981523745247, 0.07402511616736834], 'SSGradCAM': [0.07083260959921248, 0.05631198759427123, 0.0743713917093751], 'SSGradCAM Abs': [0.1138705860814457, 0.07647386142543053, 0.04327451388219104], 'Activation': [0.28083843058391755, 0.24582839969076659, 0.16065709972080947]}
# DEEPLAB = {'CRP Relevance': [0.777274751879142, 0.7615438525949279, 0.5505032161259379], 'LRP': [0.7558212798620206, 0.755818715020963, 0.5618686803339367], 'Guided GradCAM': [0.33737022397886607, 0.3828471295078427, 0.3507946950894765], 'GradCAM': [0.232634259227688, 0.2172295339651722, 0.19672131049052258], 'SSGradCAM': [0.2640619324401614, 0.32425665777826196, 0.25991240515669717], 'Activation': [0.7364490419903797, 0.7263918501436611, 0.46494091028547796]}# YOLOv5 = {'CRP Relevance': [0.020515245071324024, 0.031005894270548028, 0.022852148331988887], 'LRP': [0.030216201546139593, 0.030567902965446786, 0.09495701762105178], 'Guided GradCAM': [0.05756882938879573, 0.05293837594446657, 0.08807674911922093], 'Guided GradCAM Abs': [0.044861026271167045, 0.04198537040044782, 0.030329523485580375], 'GradCAM': [0.06306023447529552, 0.05916558944624149, 0.08009302594748485], 'SSGradCAM': [0.06158120974096802, 0.0558284155257171, 0.14577527206890994], 'SSGradCAM Abs': [0.03800880940546403, 0.03581083334809259, 0.0684975544625352], 'Activation': [0.21233562212046506, 0.15076196339801629, 0.07909097062436954]}
# scores = [YOLOv6, YOLOv5, UNET, DEEPLAB]
# for k in YOLOv5:
#     print(k)
#     print(np.mean([s[k] for s in scores]))

if __name__ == "__main__":
    main()
