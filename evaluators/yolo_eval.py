import torch

preds = torch.load(
    "src/gdrnpp/output/yolox/bop_pbr/yolox_x_1920_augCozyAAEhsv_ranger_30_epochs_mydataset_pbr_mydataset_test_primesense/model_final.pth",
    map_location="cpu"
)

