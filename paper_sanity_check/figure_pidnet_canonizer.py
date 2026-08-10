from LCRP.models import get_model
from LCRP.utils.crp_configs import ATTRIBUTORS, CANONIZERS, VISUALIZATIONS, COMPOSITES
import torch
from LCRP.utils.pidnet_canonizers import PIDNetCanonizer, PIDNetBaseCanonizer, EpsilonPlusFlatBasePIDNet, EpsilonPlusFlatforPIDNet, EpsilonPlusFlatMulforPIDNet
import numpy as np
import zennit
from torchvision import transforms
from src.datasets.flood_dataset import FloodDataset
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from paper_sanity_check.test_pidnet_canonizer import LRP, test_pidnet_canonizer

# --- device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# --- model / checkpoint
model_name = "pidnet"
ckpt_path = "./models/flood_model.pt"
data_root = "../../Datasets/flood_segmentation"
# ========= HACK: prevent get_model()/get_pidnet() from trying to strictly load its OWN checkpoint =========
# Some internal path calls `model.load_state_dict(torch.load(cfg["ckpt_path"]))` with strict=True,
# which raises due to "model." prefix or different head. We temporarily force:
#   - torch.load -> always map to CPU (safe on CPU-only)
#   - nn.Module.load_state_dict -> always use strict=False
_orig_torch_load = torch.load
_orig_load_state_dict = torch.nn.Module.load_state_dict

def _cpu_load(*args, **kwargs):
    kwargs.setdefault("map_location", device)
    return _orig_torch_load(*args, **kwargs)

def _lenient_load_state_dict(self, state_dict, strict=True):
    # Force non-strict to avoid internal RuntimeError during model construction
    return _orig_load_state_dict(self, state_dict, strict=False)

torch.load = _cpu_load
torch.nn.Module.load_state_dict = _lenient_load_state_dict
try:
    # Build the model; any internal checkpoint load will be lenient and won't crash
    model = get_model(model_name=model_name, device=device, classes=2)
finally:
    # Restore patched functions
    torch.load = _orig_torch_load
    torch.nn.Module.load_state_dict = _orig_load_state_dict
model.to(device)
model.eval()
# ========= Now load YOUR checkpoint properly (handle "model." / "module." prefixes) =========
def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state", "model", "net", "module"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj

def _strip_prefix(sd, prefix):
    if any(k.startswith(prefix) for k in sd.keys()):
        return {k[len(prefix):]: v for k, v in sd.items()}
    return sd
raw_sd = torch.load(ckpt_path, map_location=device)
sd = _extract_state_dict(raw_sd)
sd = _strip_prefix(sd, "model.")
sd = _strip_prefix(sd, "module.")

missing, unexpected = model.load_state_dict(sd, strict=False)
if missing:
    print("[load_state_dict] Missing keys:", len(missing))
    # print(missing)  # uncomment for full list
if unexpected:
    print("[load_state_dict] Unexpected keys:", len(unexpected))
    # print(unexpected)  # uncomment for full list


root_dir = "../../Datasets/flood_segmentation/"
dataset = FloodDataset(root_dir=root_dir, split="train", transform=transforms.ToTensor())

print('Loaded dataset:', type(dataset))
print('len(dataset) =', len(dataset))

ldr=torch.utils.data.DataLoader(dataset,batch_size=32)
save_dir_base="./pidnet_output/"


from LCRP.utils.crp_configs import ATTRIBUTORS, CANONIZERS, VISUALIZATIONS, COMPOSITES
import copy
from PIL import Image

def undo_to_tensor(tensor):
    images=[]

    for t in tensor:
        # Ensure tensor is detached from computation graph and moved to CPU
        t = t.detach().cpu()

        # Clamp to [0,1] in case of small floating point error
        t = torch.clamp(t, 0, 1)

        # Convert from [C, H, W] to [H, W, C] and scale to [0, 255]
        array = t.permute(1, 2, 0).numpy() * 255

        # Convert to uint8 for image representation
        array = array.astype(np.uint8)

        # Convert to PIL Image
        image = Image.fromarray(array)
        images.append(image)
    if len(images) == 1:
        images = images[0]
    return images


test_pidnet_canonizer(model, dataset, device, N=5, output=True)
for k in range(50):
    N=5
    for canonizer_str, canonizer_cls in [("EpsilonPlusFlatforPIDNet", EpsilonPlusFlatforPIDNet)]:
        plt.subplots(N,3, figsize=(10, 15))
        for i in range(N):
            x = dataset[k*N+i][0].unsqueeze(0).to(device)
            x.requires_grad_()
            # Without canonizer
            c=PIDNetBaseCanonizer(recursive=True)
            h=c.apply(model)

            out_plain = model(x)[1]
            xpl_plain = LRP(x, model, canonizer_cls)
            for handle in h:
                handle.remove()
            plt.subplot(N,3,3*i+1)
            if i==0:
                plt.title("Input")
            plt.imshow(undo_to_tensor(x))
            plt.axis('off')  
            plt.subplot(N,3,3*i+2)
            if i==0:
                plt.title("w.o. canonizer")
            plt.imshow(xpl_plain)
            plt.axis('off')

            c=PIDNetCanonizer()
            h=c.apply(model)
            out_canon = model(x)[1]
            xpl_canon = LRP(x, model, canonizer_cls)
            plt.subplot(N,3,3*i+3)
            if i==0:
                plt.title(f"w. canonizer")
            plt.imshow(xpl_canon)
            plt.axis('off')  
            for handle in h:
                handle.remove()
            out_canon = model(x)[1]
            print("\n\n\n===\n\n")
        plt.savefig(f"figure-{N}_{k}_{canonizer_str}.png", dpi=200, bbox_inches="tight")
        plt.close()
