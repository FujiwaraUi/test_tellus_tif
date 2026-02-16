import numpy as np
import timm
import os
import torch
import tifffile as tif
import segmentation_models_pytorch as smp
from IPython.display import display
from torchview import draw_graph

# IPython があれば display を使う。なければダミーにする。
try:
    from IPython.display import display  # Notebook用
except ModuleNotFoundError:
    def display(*args, **kwargs):
        return  # スクリプト実行では何もしない


def prac_resnet(hyper_cube, outdir="./data"):
    H, W, B = hyper_cube.shape

    x = hyper_cube.astype(np.float32, copy=False)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)

    inputs = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)

    cnn = timm.create_model("resnet18", pretrained=False, in_chans=B)
    print(cnn)

    outputs = cnn(inputs)[0]
    print(f"\n Model Output Class: {outputs.shape}")

    model_graph = draw_graph(cnn, input_size=tuple(inputs.shape), expand_nested=False)

    # Notebookなら表示、スクリプトなら何もしない
    display(model_graph.visual_graph)

    # 保存は常に行う
    os.makedirs(outdir, exist_ok=True)

    model_graph.visual_graph.graph_attr["dpi"] = "1800"  # 例: 300, 600, 1200 など
    model_graph.visual_graph.render(
        filename=os.path.join(outdir, "CNN_Graph"),
        format="png",
        cleanup=True,
    )
    
def prac_unet(hyper_cube, outdir="./data"):
    H, W, B = hyper_cube.shape

    x = hyper_cube.astype(np.float32, copy=False)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)

    inputs = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)

    
    unet = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=B,
        classes=10,
    )
    print(unet)
    
    outputs = unet(inputs)[0]
    print(f"\n Model Output Class: {outputs.shape}")
    
    model_graph = draw_graph(unet, input_size=tuple(inputs.shape), expand_nested=True)
    display(model_graph.visual_graph)
    
    model_graph.visual_graph.graph_attr["dpi"] = "1800"  # 例: 300, 600, 1200 など
    os.makedirs(outdir, exist_ok=True)
    model_graph.visual_graph.render(
        filename=os.path.join(outdir, "UNet_Graph"),
        format="png",
        cleanup=True,
    )
    
    
    
if __name__ == "__main__":
    filepath = "/Volumes/ssd/HSID/HISUI/HSHL1G_N329E1299_20230523072720_20240308144532.tif"
    outdir = "./data"
    hyper_cube = tif.imread(filepath)
    
    # prac_resnet(hyper_cube, outdir=outdir)

    prac_unet(hyper_cube, outdir=outdir)
