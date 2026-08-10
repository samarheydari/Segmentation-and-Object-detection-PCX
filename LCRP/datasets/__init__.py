import importlib
from typing import Dict, Any


def _load_callable(module_path: str, attr_name: str):
    # Lazy imports prevent unrelated dataset backends from failing global import.
    # This is important when only running Flood+PIDNet experiments.
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


DATASETS = {
    "coco2017": {
        "train": ("datasets.coco2017", "coco2017_train"),
        "test": ("datasets.coco2017", "coco2017_test"),
        "n_classes": 80,
    },
    "coco": {
        "train": ("datasets.coco", "coco_train"),
        "test": ("datasets.coco", "coco_test"),
        "n_classes": 81,
    },
    "cityscapes": {
        "train": ("datasets.cityscapes", "cityscapes_train"),
        "test": ("datasets.cityscapes", "cityscapes_test"),
        "n_classes": 20,
    },
    "voc2012": {
        "train": ("datasets.voc2012", "voc2012_train"),
        "test": ("datasets.voc2012", "voc2012_test"),
        "n_classes": 21,
    },
    "flood": {
        "train": ("datasets.flood", "flood_train"),
        "test": ("datasets.flood", "flood_test"),
        "n_classes": 2,
    },
    "person_car": {
        "train": ("datasets.person_car", "person_car_train"),
        "test": ("datasets.person_car", "person_car_test"),
        "n_classes": 2,
    },
}


def get_dataset(dataset_name: str) -> Dict[str, Any]:
    print("Initialize dataset:", dataset_name)
    if dataset_name not in DATASETS:
        print(f"DATASET {dataset_name} not defined.")
        exit()

    entry = DATASETS[dataset_name]
    # Keep the original contract of get_dataset():
    # return {"train": callable, "test": callable, "n_classes": int}
    # so scripts like instance_perturbation.py remain unchanged.
    return {
        "train": _load_callable(*entry["train"]),
        "test": _load_callable(*entry["test"]),
        "n_classes": entry["n_classes"],
    }
