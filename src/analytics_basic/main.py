import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import tifffile as tif


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
    

if __name__=="__main__":
    filepath = "/Volumes/ssd/HSID/HISUI/HSHL1G_N203W1558_20230818005931_20240308010611.tif"
    outdir = "./data"
    os.makedirs(outdir, exist_ok=True)
    
    hyper_cube = tif.imread(filepath)

    # check_missing_val(hyper_cube)
    plot_histrogram_HS(hyper_cube)
    plot_histrogram_HS_rm_missing_val(hyper_cube)

    
