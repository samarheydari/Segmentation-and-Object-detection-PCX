from LCRP.models import get_model
import torch
from LCRP.utils.pidnet_canonizers import PIDNetCanonizer, PIDNetBaseCanonizer
from src.datasets.flood_dataset import FloodDataset
from torch.utils.data import Dataset
from zennit.image import imgify
import zennit
from src.datasets.flood_dataset import FloodDataset
from torch.utils.data import Dataset
from LCRP.utils.pidnet_canonizers import PIDNetCanonizer, PIDNetBaseCanonizer, EpsilonPlusFlatBasePIDNet, EpsilonPlusFlatforPIDNet, EpsilonPlusFlatMulforPIDNet
def LRP(x, model, composite_cls, output_index=1, target_class=0):
    model.zero_grad(set_to_none=True)
    if x.grad is not None:
        x.grad.zero_()
    with composite_cls().context(model) as modified_model:
    # with EpsilonPlusFlatBasePIDNet().context(model) as modified_model:
        output = modified_model(x)
        if isinstance(output, (list, tuple)):
            output = output[output_index]
        # gradient/ relevance wrt. class/output 0
        init_grad = torch.zeros_like(output)
        init_grad[:, target_class, ...] = 1.0
        output.backward(gradient=init_grad)
        attr = x.grad[0, 0, :, :].detach().cpu().numpy()
    return zennit.image.imgify(attr,
                               symmetric=True,)


def test_pidnet_canonizer(model, dataset, device, N=5, output=False):
    lrp_test_done = False
    for i in range(N):
        x = torch.from_numpy(dataset[i][0]).unsqueeze(0).to(device)
        model.eval()
        x.requires_grad_()
        # Without canonizer
        out_plain = model(x)[1]
        c=PIDNetBaseCanonizer()
        h=c.apply(model)
        out_canon = model(x)[1]
        assert torch.all(torch.isclose(out_plain - out_canon,torch.zeros_like(out_plain),atol=1e-4)), f"Base canonizer has output difference {(out_plain - out_canon).abs().max()}"
        if output:
            print(f"Output difference of half canonizer after attaching: {(out_plain - out_canon).abs().max()}")
        for handle in h:
            handle.remove()
            
        out_canon = model(x)[1]
        assert torch.all(torch.isclose(out_plain - out_canon,torch.zeros_like(out_plain),atol=1e-4)), f"Base canonizer has output difference {(out_plain - out_canon).abs().max()} after detach"
        
        if output:
            print(f"Output difference of half canonizer after detaching: {(out_plain - out_canon).abs().max()}")

        c=PIDNetCanonizer()
        h=c.apply(model)
        out_canon = model(x)[1]
        # if i==0:
        #     l=[(module.weight, module.bias, module.eps, name) for name, module in model.named_modules() if isinstance(module, torch.nn.BatchNorm2d)]
        #     moduless=list(model.named_modules())
        #     overall_bn_check = [all((all(ll[0]==torch.ones_like(ll[0])), all(ll[1]==torch.zeros_like(ll[1])), ll[2]==0.)) for ll in l]
        #     missing_bn_ids = torch.where(torch.tensor([not bn for bn in overall_bn_check]))
        #     names=[l[idx][-1] for idx in missing_bn_ids[0].tolist()]
        #     print(f"BatchNorm2d layers not handled by canonizer: {names}")
        assert torch.all(torch.isclose(out_plain - out_canon,torch.zeros_like(out_plain),atol=1e-4)), f"Full canonizer has output difference {(out_plain - out_canon).abs().max()}"
        print(f"Output difference of canonizer after attaching: {(out_plain - out_canon).abs().max()}")
        for handle in h:
            handle.remove()
        out_canon = model(x)[1]
        assert torch.all(torch.isclose(out_plain - out_canon,torch.zeros_like(out_plain),atol=1e-4)), f"Full canonizer has output difference {(out_plain - out_canon).abs().max()} after detach"
        print(f"Output difference of canonizer after detaching: {(out_plain - out_canon).abs().max()}")
        print("\n\n\n===\n\n")


class ToTensorDataset(Dataset):
    def __init__(self, ds):
        self.ds=ds
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        dp = self.ds[idx]
        return torch.from_numpy(dp[0]),*dp[1:]

if __name__=="__main__":
    # --- device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # --- model / checkpoint
    model_name = "pidnet"
    ckpt_path = "/home/heydari/paper/LCRP/models/flood_model.pt"
    data_root = "/home/heydari/FHHI-XAI/data/flood_segmentation"
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
    root_dir = "/home/heydari/FHHI-XAI/data/flood_segmentation/"
    dataset = FloodDataset(root_dir=root_dir, split="train")
    # dataset = ToTensorDataset(dataset)
    
    print('Loaded dataset:', type(dataset))
    print('len(dataset) =', len(dataset))

    ldr=torch.utils.data.DataLoader(dataset,batch_size=32)
    save_dir_base="./pidnet_output/"
    test_pidnet_canonizer(model, dataset, device, N=5, output=True)



