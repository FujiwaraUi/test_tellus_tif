import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import tifffile as tif
import math


def check_missing_val(hyper_cube):
    flat = hyper_cube.ravel()  
    # dtype から取り得る最大値を取得（uint16 なら 65535）
    info = np.iinfo(hyper_cube.dtype)
    n_bins = int(info.max) + 1  # 65536
    
    counts = np.bincount(flat, minlength=n_bins)

    # 代表値の個数をまず表示（欠損候補の当たりを付ける）
    print("count(value=0)     =", int(counts[0]))
    print("count(value=65535) =", int(counts[65535]))
    print("min value =", int(flat.min()), "count =", int(counts[int(flat.min())]))
    print("max value =", int(flat.max()), "count =", int(counts[int(flat.max())]))

    # 個数が多い値トップ20を表示
    topk = 20
    idx = np.argsort(counts)[::-1][:topk]
    print(f"Top-{topk} most frequent values:")
    for v in idx:
        c = int(counts[v])
        if c == 0:
            break
        print(f"value={int(v):5d}  count={c}")

def plot_histrogram_HS(hyper_cube):
    hc_flat = hyper_cube.ravel()

    plt.figure(figsize=(14,3))
    plt.title("Histrogram of the HS Cube")
    plt.hist(hc_flat, bins=300)
    plt.yscale("linear")
    plt.savefig(os.path.join(outdir, "histogram.png"), dpi=200)
    plt.close()

    
def plot_histrogram_HS_rm_missing_val(hyper_cube):
    flat = hyper_cube.ravel()
    counts = np.bincount(flat, minlength=65526)

    # 欠損（NoData）候補を指定(0 を想定)
    nodata_values = [0, 65535]        # 例: [0] または [0, 65535]

    counts_clean = counts.copy()
    for nv in nodata_values:
        counts_clean[nv] = 0

    values = np.arange(65536)

    plt.figure(figsize=(14, 3))
    plt.title("Histogram value counts (NoData removed), y=log")
    plt.hist(values, bins=300, weights=counts_clean)
    plt.yscale("log")  # y軸をlog
    plt.xlabel("DN value")
    plt.ylabel("count (log)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "histogram_log_nodata_removed.png"), dpi=300)
    plt.close()

def visualization_HS_cube(hyper_cube,IDX_BAND=0):
    plt.figure(figsize=(16,10), dpi=120, facecolor="w",  edgecolor="k")
    
    for i in range(24):
        print(i)
        plt.subplot(3, 8, i+1)
        plt.title('Band {}'.format(i+1))
        plt.imshow(hyper_cube[:,:,i], vmin=1, vmax=65535)
#        plt.imshow(hyper_cube[:,:,i], cmap='jet', vmin=1, vmax=185)
        plt.colorbar(shrink=0.5)
        plt.axis('off')
        plt.tight_layout()
    plt.savefig('./data/all_bands.png')
    plt.show();plt.clf();plt.close()
#    plt.title(f"Band{IDX_BAND}")
#    plt.imshow(hyper_cube[:,:,IDX_BAND], vmin=1)
#    plt.colorbar(shrink=0.6)
#    plt.tight_layout()
#    plt.savefig(os.path.join(outdir, f"visualization_HS_cube_{IDX_BAND}.png"), dpi=300)


def visualization_HS_cube_a(hyper_cube, n_show=None, outpath="./data/all_bands.png",
                            ncols=10, percentile=(1, 99), downsample=4):
    """
    hyper_cube: (H, W, B)
    n_show: 表示するバンド数（Noneなら全バンド）
    ncols: 1行あたりの枚数
    percentile: 表示レンジ決定のパーセンタイル
    downsample: 表示用に間引く（大きいほど軽い）。4なら 1/4 解像度で表示。
    """

    if hyper_cube.ndim != 3:
        raise ValueError(f"hyper_cube must be 3D (H,W,B), got shape={hyper_cube.shape}")

    H, W, B = hyper_cube.shape
    if n_show is None:
        n_show = B
    n_show = min(int(n_show), B)

    # 表示を軽くする（保存画像の見やすさも上がる）
    if downsample and downsample > 1:
        cube0 = hyper_cube[::downsample, ::downsample, :n_show]
    else:
        cube0 = hyper_cube[:, :, :n_show]

    # 欠損候補をマスク（必要ならここを調整）
    cube = cube0.astype(np.float32, copy=False)
    mask = (cube == 0) | (cube == 65535)
    cube = np.ma.array(cube, mask=mask)

    # 表示レンジを全バンド共通で決定（比較しやすい）
    valid = cube.compressed()
    if valid.size == 0:
        raise ValueError("All pixels are masked. Check missing-value rules (0 / 65535).")
    vmin, vmax = np.percentile(valid, percentile)

    # グリッド計算
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n_show / ncols))

    # 図のサイズ：1枚あたりの大きさから決める（固定値より破綻しにくい）
    fig_w = max(12, ncols * 1.6)
    fig_h = max(8,  nrows * 1.6)

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), dpi=120, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    im = None
    for i in range(n_show):
        ax = axes[i]
        im = ax.imshow(cube[:, :, i], vmin=vmin, vmax=vmax)
        ax.set_title(f"Band {i+1}", fontsize=8)
        ax.axis("off")

    # 余った枠を消す
    for j in range(n_show, len(axes)):
        axes[j].axis("off")

    # 共有カラーバー（figureに紐づける）
    fig.colorbar(im, ax=axes.tolist(), shrink=0.6)

    fig.savefig(outpath, dpi=600)
    plt.show()
    plt.close(fig)

if __name__=="__main__":
    # https://sorabatake.jp/40363/
    # filepath = "/Volumes/ssd/HSID/HISUI/HSHL1G_N203W1558_20230818005931_20240308010611.tif"
    filepath = "/Volumes/ssd/HSID/HISUI/HSHL1G_N329E1299_20230523072720_20240308144532.tif"
    outdir = "./data"
    os.makedirs(outdir, exist_ok=True)
    
    hyper_cube = tif.imread(filepath)

    print(hyper_cube.shape)

    # check_missing_val(hyper_cube)
    plot_histrogram_HS(hyper_cube)
    plot_histrogram_HS_rm_missing_val(hyper_cube)
    # visualization_HS_cube(hyper_cube)
    visualization_HS_cube_a(hyper_cube, n_show=None, ncols=10, outpath="./data/all_bands.png")

    

    
 
