import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import rasterio
from PIL import Image
from scipy.signal import convolve2d 

#  Utility Functions 

def read_band(path):
    """Read a single-band raster as float32. Return None if missing."""
    if not os.path.exists(path):
        return None
    with rasterio.open(path) as src:
        return src.read(1).astype("float32")

def get_geo(path):
    """Get geotransform and CRS from a reference raster."""
    with rasterio.open(path) as src:
        transform = tuple(src.transform.to_gdal())
        crs = src.crs.to_string() if src.crs else ""
    return transform, crs

def maybe_scale_s2_to_01(band):
    """
    Make S2 reflectance roughly 0..1 if it looks like 0..10000.
    Then hard-clip to [0,1] to keep training stable.
    """
    finite = np.isfinite(band)
    if not np.any(finite):
        return np.zeros_like(band, dtype=np.float32)
    q95 = np.nanpercentile(band[finite], 95)
    if q95 > 2.0:  # likely in 0..10000
        band = band / 10000.0
    # hard clamp to [0,1] to avoid outliers wrecking training
    band = np.clip(band, 0.0, 1.0)
    return band.astype(np.float32)

def build_mask(inputs, target, colloc=None):
    """
    Valid pixel mask: all inputs and target are finite,
    and collocationFlags > 0 if provided.
    """
    mask = np.isfinite(inputs).all(axis=0) & np.isfinite(target).all(axis=0)
    if colloc is not None:
        mask &= (colloc > 0)
    return mask

def zscore_inplace(x, mask):
    """In-place z-score normalization using stats computed on valid pixels only."""
    if mask is None or not np.any(mask):
        mu, sigma = np.nanmean(x), np.nanstd(x)
    else:
        mu, sigma = float(np.nanmean(x[mask])), float(np.nanstd(x[mask]))
    if not np.isfinite(mu):
        mu = 0.0
    if (not np.isfinite(sigma)) or sigma < 1e-6:
        sigma = 1.0
    x -= mu
    x /= sigma

def norm_to_uint8(a):
    """Normalize array to [0,255] for visualization using 2nd-98th percentile stretch."""
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    p2, p98 = np.percentile(a, [2, 98])
    if p98 - p2 < 1e-6:
        p98 = p2 + 1.0
    a = np.clip((a - p2) / (p98 - p2), 0, 1)
    return (a * 255).astype(np.uint8)

def make_rgb(b1, b2, b3):
    """Create uint8 RGB image from three bands."""
    R = norm_to_uint8(b1)
    G = norm_to_uint8(b2)
    B = norm_to_uint8(b3)
    return np.dstack([R, G, B])

def patch_iter(H, W, ps, stride):
    """Sliding window coordinates."""
    for r in range(0, H - ps + 1, stride):
        for c in range(0, W - ps + 1, stride):
            yield r, c

# Filters

def dark_fraction(Y, M, thr=0.10):
    """
    Fraction of very dark pixels within mask.
    A pixel is 'dark' if mean(B2,B3,B4) < thr AND B8 < thr.
    thr default = 0.10 (stricter than 0.03).
    """
    if not np.any(M):
        return 1.0
    vis_mean = (Y[0] + Y[1] + Y[2]) / 3.0
    dark = (vis_mean < thr) & (Y[3] < thr) & M
    return float(dark.sum()) / float(M.sum())

def laplacian_var(img, M):
    """
    Laplacian variance (texture) inside mask for one channel (use B8).
    """
    k = np.array([[0, 1, 0],
                  [1,-4, 1],
                  [0, 1, 0]], dtype=np.float32)
    a = img.copy()
    # Fill invalids to local mean to avoid NaN propagating
    bad = ~np.isfinite(a)
    if np.any(bad & M):
        meanv = np.nanmean(a[M])
        a[bad] = meanv
    L = convolve2d(a, k, mode="same", boundary="symm")
    return float(np.nanvar(L[M])) if np.any(M) else 0.0

# Main 

def main(args):
    base_dir = args.base_dir
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    preview_dir = os.path.join(out_dir, "preview_patches")
    os.makedirs(preview_dir, exist_ok=True)

    # Only take leaf folders (skip preview dirs if any)
    folders = sorted([
        f for f in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, f))
    ])

    # Distribute max_patches across folders if requested
    per_folder_cap = None
    if args.max_patches and args.per_folder_share and len(folders) > 0:
        per_folder_cap = max(1, args.max_patches // len(folders))

    count = 0
    manifest = []

    # Stats
    dark_skipped = 0
    texture_skipped = 0
    validratio_skipped = 0
    var_skipped = 0

    pbar = tqdm(folders, desc="Processing folders")
    for folder in pbar:
        if args.max_patches and count >= args.max_patches:
            break
        folder_path = os.path.join(base_dir, folder)

        # Read S2 target bands (B2, B3, B4, B8)
        target_band_names = ["B2.img", "B3.img", "B4.img", "B8.img"]
        target_bands = []
        for name in target_band_names:
            band = read_band(os.path.join(folder_path, name))
            if band is None:
                target_bands = []
                break
            target_bands.append(maybe_scale_s2_to_01(band))
        if not target_bands:
            continue
        target = np.stack(target_bands, axis=0).astype(np.float32)  # (4, H, W)
        transform, crs = get_geo(os.path.join(folder_path, "B2.img"))
        H, W = target.shape[1], target.shape[2]

        # Read S1 inputs
        def try_read_s1(names):
            arrs = [read_band(os.path.join(folder_path, n)) for n in names]
            return None if any(a is None for a in arrs) else np.stack(arrs, 0)

        s1 = try_read_s1(["Sigma0_HH_db_m.img", "Sigma0_HV_db_m.img"])
        if s1 is None:
            s1 = try_read_s1(["Sigma0_HH_db_corr024_m.img", "Sigma0_HV_db_corr024_m.img"])
        if s1 is None:
            s1 = try_read_s1(["Sigma0_HH_db_corr028_m.img", "Sigma0_HV_db_corr028_m.img"])
        if s1 is None:
            continue

        # Read auxiliary layers
        aux_paths = ["projectedLocalIncidenceAngle_m.img", "elevation_ref_egm2008.img"]
        aux = [read_band(os.path.join(folder_path, p)) for p in aux_paths]
        if any(a is None for a in aux):
            continue
        aux = np.stack(aux, 0).astype(np.float32)

        inputs = np.concatenate([s1.astype(np.float32), aux], axis=0)  # (4,H,W)

        # Collocation mask
        colloc_path = os.path.join(folder_path, "collocationFlags.img")
        colloc = read_band(colloc_path) if os.path.exists(colloc_path) else None
        valid_mask = build_mask(inputs, target, colloc)

        saved_here = 0
        for row, col in patch_iter(H, W, args.patch_size, args.stride):
            if args.max_patches and count >= args.max_patches:
                break
            if per_folder_cap and saved_here >= per_folder_cap:
                break

            X = inputs[:, row:row + args.patch_size, col:col + args.patch_size].copy()
            Y = target[:, row:row + args.patch_size, col:col + args.patch_size].copy()
            M = valid_mask[row:row + args.patch_size, col:col + args.patch_size].copy()

            # 1) Valid ratio
            vr = float(M.mean()) if M.size > 0 else 0.0
            if vr < args.valid_ratio_threshold:
                validratio_skipped += 1
                continue

            # 2) Variance check on targets (all channels very flat -> skip)
            if all(np.nanvar(Y[ch][M]) < args.variance_threshold for ch in range(Y.shape[0])):
                var_skipped += 1
                continue

            # 3) Dark fraction filter (stricter)
            if dark_fraction(Y, M, thr=args.dark_thr) > args.dark_max_ratio:
                dark_skipped += 1
                continue

            # 4) Texture filter on B8 (stronger)
            if laplacian_var(Y[3], M) < args.texture_thr:
                texture_skipped += 1
                continue

            # Normalize + sanitize
            # S1 HH/HV: z-score using valid mask
            zscore_inplace(X[0], M)  # HH
            zscore_inplace(X[1], M)  # HV
            # Angles/Elevation: simple scaling to reasonable ranges
            X[2] = np.nan_to_num(X[2], nan=0.0) / 90.0     # IncAngle in ~[0,90]
            X[3] = np.nan_to_num(X[3], nan=0.0) / 1000.0   # Elevation in km

            # Targets Y: already clipped to [0,1] above; fill invalids with 0
            # Inputs X: also fill invalids with 0 (outside mask) to avoid NaN leakage
            for ch in range(X.shape[0]):
                ch_arr = X[ch]
                ch_arr[~M] = 0.0
                X[ch] = np.nan_to_num(ch_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            for ch in range(Y.shape[0]):
                ch_arr = Y[ch]
                ch_arr[~M] = 0.0
                Y[ch] = np.nan_to_num(ch_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            # Save NPZ
            patch_id = f"{count:06d}"
            out_npz = os.path.join(out_dir, f"patch_{patch_id}.npz")
            meta = dict(folder=folder, row=int(row), col=int(col),
                        transform=list(transform), crs=str(crs),
                        patch_size=args.patch_size, stride=args.stride,
                        valid_ratio=float(vr))
            np.savez_compressed(out_npz,
                                inputs=X, target=Y, mask=M.astype("uint8"),
                                **meta)

            # Save previews
            patch_prev_dir = os.path.join(preview_dir, f"patch_{patch_id}")
            os.makedirs(patch_prev_dir, exist_ok=True)

            ch_names_in = ["HH_dB_std", "HV_dB_std", "IncAngle_n", "Elevation_km"]
            for name, arr in zip(ch_names_in, X):
                Image.fromarray(norm_to_uint8(arr)).save(os.path.join(patch_prev_dir, f"{name}.png"))
            ch_names_tar = ["B2", "B3", "B4", "B8"]
            for name, arr in zip(ch_names_tar, Y):
                Image.fromarray(norm_to_uint8(arr)).save(os.path.join(patch_prev_dir, f"{name}.png"))

            # true color & CIR (use targets)
            rgb_true = make_rgb(Y[2], Y[1], Y[0])  # B4,B3,B2
            Image.fromarray(rgb_true).save(os.path.join(patch_prev_dir, "true_color.png"))
            rgb_cir = make_rgb(Y[3], Y[2], Y[1])   # B8,B4,B3
            Image.fromarray(rgb_cir).save(os.path.join(patch_prev_dir, "false_color_CIR.png"))

            manifest.append({
                "patch_id": patch_id,
                "folder": folder,
                "npz": os.path.relpath(out_npz, out_dir),
                "preview_dir": os.path.relpath(patch_prev_dir, out_dir),
                "row": int(row), "col": int(col),
                "valid_ratio": float(vr)
            })

            count += 1
            saved_here += 1

        pbar.set_postfix(saved=count)

    # Save manifest
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({
            "total_patches": count,
            "dark_skipped": dark_skipped,
            "texture_skipped": texture_skipped,
            "validratio_skipped": validratio_skipped,
            "var_skipped": var_skipped,
            "base_dir": base_dir,
            "patch_size": args.patch_size,
            "stride": args.stride,
            "valid_ratio_threshold": args.valid_ratio_threshold,
            "variance_threshold": args.variance_threshold,
            "dark_thr": args.dark_thr,
            "dark_max_ratio": args.dark_max_ratio,
            "texture_thr": args.texture_thr,
            "patches": manifest[:2000]
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done! Saved {count} patches")
    print(f"   Skipped: valid_ratio={validratio_skipped}, dark={dark_skipped}, low_texture={texture_skipped}, low_var={var_skipped}")
    print(f"   Training data: {out_dir}")
    print(f"   Previews: {preview_dir}")

# CLI

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True,
                    help="Directory containing multiple *_collocated.data folders")
    ap.add_argument("--output-dir", required=True,
                    help="Output directory for npz and preview images")
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--max-patches", type=int, default=10000)
    ap.add_argument("--per-folder-share", action="store_true",
                    help="Distribute patches evenly across folders")

    # Filtering thresholds (tightened defaults)
    ap.add_argument("--valid-ratio-threshold", type=float, default=0.80,
                    help="Min fraction of valid pixels inside a patch")
    ap.add_argument("--variance-threshold", type=float, default=1e-4,
                    help="Skip patch if ALL target bands have var < this (on valid pixels)")
    ap.add_argument("--dark-thr", type=float, default=0.10,
                    help="A pixel is 'dark' if mean(B2..B4)<thr AND B8<thr (reflectance)")
    ap.add_argument("--dark-max-ratio", type=float, default=0.60,
                    help="Skip patch if dark pixel fraction > this")
    ap.add_argument("--texture-thr", type=float, default=5e-5,
                    help="Min Laplacian variance on B8 within mask")

    args = ap.parse_args()
    main(args)
