import os
import numpy as np
import tifffile as tif
import cv2
import plotly.graph_objects as go

def visualize_hsc_3d(hyper_cube, outdir):
    PATCH = 64
    RESIZE = 16

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "data")
    os.makedirs(output_dir, exist_ok=True)

    print(hyper_cube.shape, type(hyper_cube))
    # (1821, 1859, 185) <class 'numpy.ndarray'>
    
    hyper_cube_patch = hyper_cube[400:400+PATCH, 200:200+PATCH, :]
    np.save(os.path.join(current_dir, f'hypercube_patch_{PATCH}.npy'), hyper_cube_patch)
    hyper_cube_patch = np.load(os.path.join(current_dir, f'hypercube_patch_{PATCH}.npy'))

    hyper_cube_patch = cv2.resize(hyper_cube_patch, (RESIZE, RESIZE), interpolation=cv2.INTER_CUBIC)

    hyper_cube_patch = np.concatenate([hyper_cube_patch, np.zeros((RESIZE, RESIZE, 1))], axis=2)


    H, W, B = hyper_cube_patch.shape
    X, Y, Z = np.mgrid[0:H, 0:W, 0:B]

    img_rgb = hyper_cube_patch[:,:,[2]]

    img_rgb = hyper_cube_patch[:, :, [2]]

    print(f'Drawing hypercube patch with shape {hyper_cube_patch.shape}')

    # plot 3D volume
    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=hyper_cube_patch.flatten(),
        opacity=0.2, # needs to be small to see through all surfaces
        surface_count=17, # needs to be a large number for good volume rendering
        colorscale="Jet",  # カラーマップ Viridis, Cividis, Inferno, Magma, Plasma, Turbo, Jet
        ))

    fig.add_trace(go.Surface(
        x=np.arange(H),
        y=np.arange(W),
        z=np.full((H, W), B),  # Z=B-1 の位置に配置
        surfacecolor=img_rgb,  # グレースケール画像を適用
        colorscale="Viridis",  # 画像のカラーマップ
        opacity=0.9, # needs to be small to see through all surfaces
    ))

    # 軸の設定 & 高解像度
    fig.update_layout(
        title="Hyperspectral Data 3D Visualization",
        width=1200,  # 高解像度用にサイズアップ
        height=800,
        scene=dict(
            xaxis_title="Height [Spacial]",
            yaxis_title="Width [Spacial]",
            zaxis_title="Spectral [Band]",
            aspectmode="cube"
        )
    )
    fig.show();
    # save image
    fig.write_image(os.path.join(current_dir, 'hypercube_patch.png'), scale=2)

    
    
if __name__ == "__main__":
    filepath = "/Volumes/ssd/HSID/HISUI/HSHL1G_N329E1299_20230523072720_20240308144532.tif"
    outdir = "./data"
    hyper_cube = tif.imread(filepath)

    visualize_hsc_3d(hyper_cube, outdir)
