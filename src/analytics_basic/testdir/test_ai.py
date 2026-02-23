import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tif
import cv2
import plotly.graph_objects as go


def bincount_cube_uint(cube: np.ndarray) -> np.ndarray:
    """
    3D cube (H, W, B) の全要素について値の頻度 counts を返す。
    ravel() 一発だと巨大データで重いので band ごとに加算する。
    """
    info = np.iinfo(cube.dtype)
    n_bins = int(info.max) + 1
    H, W, B = cube.shape

    counts = np.zeros(n_bins, dtype=np.int64)
    for k in range(B):
        flat = cube[:, :, k].reshape(-1).astype(np.int64, copy=False)
        counts += np.bincount(flat, minlength=n_bins).astype(np.int64, copy=False)
    return counts


def check_missing_val(hyper_cube: np.ndarray, topk: int = 20, assume_nodata=(0, 65535)):
    """
    代表値の個数と頻度上位を表示し、assume_nodata を欠損値として返す。
    """
    info = np.iinfo(hyper_cube.dtype)
    n_bins = int(info.max) + 1

    counts = bincount_cube_uint(hyper_cube)

    # 代表値
    vmin = int(np.min(hyper_cube))
    vmax = int(np.max(hyper_cube))

    print("count(value=0)     =", int(counts[0]) if 0 < n_bins else 0)
    if 65535 < n_bins:
        print("count(value=65535) =", int(counts[65535]))
    else:
        print("count(value=65535) = (not in dtype range)")

    print("min value =", vmin, "count =", int(counts[vmin]))
    print("max value =", vmax, "count =", int(counts[vmax]))

    idx = np.argsort(counts)[::-1][:topk]
    print(f"Top-{topk} most frequent values:")
    for v in idx:
        c = int(counts[v])
        if c == 0:
            break
        print(f"value={int(v):5d}  count={c}")

    # 欠損値を確定（あなたの方針通り）
    nodata_values = []
    for nv in assume_nodata:
        if 0 <= int(nv) < n_bins:
            nodata_values.append(int(nv))

    return counts, nodata_values


def plot_histrogram_HS(hyper_cube: np.ndarray, outdir: str, bins: int = 300):
    """
    全体ヒストグラム（counts を作ってから描画する。巨大データでも落ちにくい）
    """
    os.makedirs(outdir, exist_ok=True)
    counts = bincount_cube_uint(hyper_cube)

    n_bins = counts.size
    values = np.arange(n_bins)

    plt.figure(figsize=(14, 3))
    plt.title("Histogram of the HS cube (counts)")
    plt.hist(values, bins=bins, weights=counts)
    plt.yscale("linear")
    plt.xlabel("DN value")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "histogram.png"), dpi=200)
    plt.close()


def plot_histrogram_HS_rm_missing_val(hyper_cube: np.ndarray, outdir: str, nodata_values=(0, 65535), bins: int = 300):
    """
    欠損値を 0 にして除外したヒストグラム（y=log）
    minlength の誤り(65526)を修正し、dtype から bins 数を決める。
    """
    os.makedirs(outdir, exist_ok=True)

    counts = bincount_cube_uint(hyper_cube)
    n_bins = counts.size

    counts_clean = counts.copy()
    for nv in nodata_values:
        nv = int(nv)
        if 0 <= nv < n_bins:
            counts_clean[nv] = 0

    values = np.arange(n_bins)

    plt.figure(figsize=(14, 3))
    plt.title("Histogram (NoData removed), y=log")
    plt.hist(values, bins=bins, weights=counts_clean)
    plt.yscale("log")
    plt.xlabel("DN value")
    plt.ylabel("count (log)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "histogram_log_nodata_removed.png"), dpi=300)
    plt.close()


def visualization_HS_cube(hyper_cube: np.ndarray, outdir: str, IDX_BAND: int = 0, nodata_values=(0, 65535)):
    """
    2D 表示。欠損値をマスクし、パーセンタイルで表示レンジを決める。
    """
    os.makedirs(outdir, exist_ok=True)

    band = hyper_cube[:, :, IDX_BAND].astype(np.float32, copy=False)
    mask = np.zeros_like(band, dtype=bool)
    for nv in nodata_values:
        mask |= (band == float(nv))

    valid = band[~mask]
    if valid.size > 0:
        vmin, vmax = np.percentile(valid, [2, 98])
    else:
        vmin, vmax = 0.0, 1.0

    plt.figure(figsize=(16, 10), dpi=120, facecolor="w", edgecolor="k")
    plt.title(f"Band{IDX_BAND}")
    plt.imshow(np.ma.array(band, mask=mask), vmin=vmin, vmax=vmax)
    plt.colorbar(shrink=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"visualization_HS_cube_{IDX_BAND}.png"), dpi=300)
    plt.close()


def resize_cube_per_band(cube: np.ndarray, out_h: int, out_w: int, interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    """
    (H, W, B) をバンドごとに2Dリサイズして (out_h, out_w, B) を返す
    """
    h, w, b = cube.shape
    cube_f = cube.astype(np.float32, copy=False)
    out = np.empty((out_h, out_w, b), dtype=np.float32)
    for k in range(b):
        out[:, :, k] = cv2.resize(cube_f[:, :, k], (out_w, out_h), interpolation=interpolation)
    return out


def pick_patch_origin_valid(hyper_cube: np.ndarray, patch: int, nodata_values=(0, 65535), ref_band: int = None):
    """
    有効領域からパッチ中心を推定して r0, c0 を返す。
    参照バンド ref_band を1つ選び、そのバンドで欠損値でない画素を有効とする。
    """
    H, W, B = hyper_cube.shape
    if ref_band is None:
        ref_band = B // 2
    ref_band = int(np.clip(ref_band, 0, B - 1))

    band = hyper_cube[:, :, ref_band]
    valid = np.ones((H, W), dtype=bool)
    for nv in nodata_values:
        valid &= (band != nv)

    ys, xs = np.where(valid)
    if ys.size == 0:
        return 0, 0

    cy = int(np.median(ys))
    cx = int(np.median(xs))

    r0 = max(0, min(cy - patch // 2, H - patch))
    c0 = max(0, min(cx - patch // 2, W - patch))
    return r0, c0


def normalize_patch_excluding_nodata(cube_patch: np.ndarray, nodata_values=(0, 65535), p_low=2, p_high=98) -> np.ndarray:
    """
    欠損値を除外してパーセンタイル正規化し 0..1 にする。
    欠損位置は 0 に置く（Volume 的には透明になりやすい）。
    """
    cube = cube_patch.astype(np.float32, copy=False)

    valid = np.ones(cube.shape, dtype=bool)
    for nv in nodata_values:
        valid &= (cube != float(nv))

    vals = cube[valid]
    if vals.size == 0:
        return np.zeros_like(cube, dtype=np.float32)

    lo, hi = np.percentile(vals, [p_low, p_high])
    out = np.zeros_like(cube, dtype=np.float32)
    out[valid] = np.clip((cube[valid] - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    return out


def visualize_hsc_3d(hyper_cube: np.ndarray, outdir: str, patch=64, resize=16, plane_band=2):
    os.makedirs(outdir, exist_ok=True)

    # ここで欠損値を判断（あなたの方針に合わせて 0 と 65535 を採用）
    _, nodata_values = check_missing_val(hyper_cube, topk=20, assume_nodata=(0, 65535))
    print("assumed nodata_values =", nodata_values)

    # 有効領域からパッチを選ぶ（固定 (400,200) を廃止）
    r0, c0 = pick_patch_origin_valid(hyper_cube, patch=patch, nodata_values=nodata_values, ref_band=plane_band)
    cube_patch = hyper_cube[r0:r0 + patch, c0:c0 + patch, :]

    # 保存しておく
    np.save(os.path.join(outdir, f"hypercube_patch_{patch}.npy"), cube_patch)

    # リサイズ
    cube_patch = resize_cube_per_band(cube_patch, resize, resize, interpolation=cv2.INTER_LINEAR)

    # 欠損値を除外して正規化（ここが 3D 単色回避の主役）
    cube_vis = normalize_patch_excluding_nodata(cube_patch, nodata_values=nodata_values, p_low=2, p_high=98)

    H, W, B = cube_vis.shape
    plane_band = int(np.clip(plane_band, 0, B - 1))
    plane = cube_vis[:, :, plane_band]  # (H, W) 0..1

    X, Y, Z = np.mgrid[0:H, 0:W, 0:B]

    print("3D patch shape:", cube_vis.shape, "origin:", (r0, c0), "plane_band:", plane_band)
    print("patch nodata ratio (approx):",
          float(np.mean(np.isclose(cube_patch, nodata_values[0]))) if len(nodata_values) else 0.0)

    fig = go.Figure(data=go.Volume(
        x=X.ravel(),
        y=Y.ravel(),
        z=Z.ravel(),
        value=cube_vis.ravel(),
        isomin=0.05,
        isomax=0.95,
        opacity=0.15,
        surface_count=25,
        colorscale="Turbo",
        caps=dict(x_show=False, y_show=False, z_show=False),
    ))

    # 上面に 2D を貼る（Plotly の都合で surfacecolor は転置して合わせる）
    fig.add_trace(go.Surface(
        x=np.arange(H),
        y=np.arange(W),
        z=np.full((W, H), B - 0.5),
        surfacecolor=plane.T,
        colorscale="Viridis",
        opacity=0.95,
        showscale=False,
    ))

    fig.update_layout(
        title="Hyperspectral Data 3D Visualization",
        width=1200,
        height=800,
        scene=dict(
            xaxis_title="Height (row index)",
            yaxis_title="Width (col index)",
            zaxis_title="Spectral (band index)",
            aspectmode="cube"
        )
    )

    # HTML は確実に保存できる
    fig.write_html(os.path.join(outdir, "hypercube_patch.html"), include_plotlyjs="cdn")

    # PNG は kaleido が必要
    try:
        fig.write_image(os.path.join(outdir, "hypercube_patch.png"), scale=2)
    except Exception as e:
        print("write_image failed. PNG が必要なら: pip install -U kaleido")
        print("Error:", repr(e))

    fig.show()


if __name__ == "__main__":
    filepath = "/Volumes/ssd/HSID/HISUI/HSHL1G_N329E1299_20230523072720_20240308144532.tif"
    outdir = "./data"
    hyper_cube = tif.imread(filepath)

    # 欠損チェックとヒストグラム
    counts, nodata_values = check_missing_val(hyper_cube, topk=20, assume_nodata=(0, 65535))
    plot_histrogram_HS(hyper_cube, outdir)
    plot_histrogram_HS_rm_missing_val(hyper_cube, outdir, nodata_values=nodata_values)

    # 2D バンド表示
    visualization_HS_cube(hyper_cube, outdir, IDX_BAND=78, nodata_values=nodata_values)

    # 3D 可視化
    visualize_hsc_3d(hyper_cube, outdir, patch=64, resize=16, plane_band=78)
 

