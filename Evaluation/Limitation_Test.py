import os
import math
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Schedules

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = T + 1
    t = torch.linspace(0, T, steps, dtype=torch.float64)
    f = torch.cos(((t / T + s) / (1 + s)) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clip(betas, 1e-5, 0.999).float()

def linear_beta_schedule(T: int, beta_start=1e-4, beta_end=0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)

def make_schedule(T: int, kind: str) -> torch.Tensor:
    if kind == "cosine":
        return cosine_beta_schedule(T)
    elif kind == "linear":
        return linear_beta_schedule(T)
    else:
        raise ValueError(f"Unknown time_schedule: {kind}")


# Dataset

class S1toS2Dataset(Dataset):
    """Loads .npz patches with inputs, target, mask."""
    def __init__(self, patch_dir: str, max_files=None):
        files = [
            os.path.join(patch_dir, f)
            for f in os.listdir(patch_dir)
            if f.endswith(".npz") and os.path.isfile(os.path.join(patch_dir, f))
        ]
        files.sort()
        if max_files is not None:
            files = files[:max_files]
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        x_cond = np.nan_to_num(data["inputs"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        x_tgt  = np.nan_to_num(data["target"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        mask   = None
        if "mask" in data:
            mask = np.nan_to_num(data["mask"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(x_cond), torch.from_numpy(x_tgt), (
            torch.from_numpy(mask) if mask is not None else None
        )



# Model (U-Net Small)

class UNetSmall(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, base_ch: int = 96):
        super().__init__()
        def conv_block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.ReLU(inplace=False),
            )
        self.inc   = nn.Sequential(nn.Conv2d(in_ch + 1, base_ch, 3, padding=1), nn.ReLU(inplace=False))
        self.down1 = nn.Sequential(conv_block(base_ch,   base_ch*2), nn.MaxPool2d(2))
        self.down2 = nn.Sequential(conv_block(base_ch*2, base_ch*4), nn.MaxPool2d(2))
        self.down3 = nn.Sequential(conv_block(base_ch*4, base_ch*8), nn.MaxPool2d(2))

        self.up3   = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.conv3 = conv_block(base_ch*8, base_ch*4)

        self.up2   = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.conv2 = conv_block(base_ch*4, base_ch*2)

        self.up1   = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.conv1 = conv_block(base_ch*2, base_ch)

        self.outc  = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, xt_and_cond, t_idx):
        B, _, H, W = xt_and_cond.shape
        # IMPORTANT: integer timestep channel (same as training). Do NOT normalize.
        t_map = t_idx.view(B, 1, 1, 1).float().repeat(1, 1, H, W)
        x = torch.cat([xt_and_cond, t_map], dim=1)
        e1 = self.inc(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        d3 = self.up3(e4); d3 = torch.cat([d3, e3], dim=1); d3 = self.conv3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, e2], dim=1); d2 = self.conv2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, e1], dim=1); d1 = self.conv1(d1)
        return self.outc(d1)



# Mask & Metrics (pixel-weighted)

def _ensure_mask(mask, B, H, W, device, dtype=torch.float32):
    if mask is None:
        return torch.ones((B, 1, H, W), device=device, dtype=dtype)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    return (mask > 0).to(dtype).to(device)

@torch.no_grad()
def channelwise_error_sums(pred, tgt, mask=None):
    """
    Return:
      abs_sum_c: (C,) sum of |pred - tgt| over masked pixels and batch
      sq_sum_c:  (C,) sum of (pred - tgt)^2 over masked pixels and batch
      pix_sum:   scalar, total number of valid pixels (per-channel identical)
    """
    B, C, H, W = pred.shape
    device = pred.device
    w = _ensure_mask(mask, B, H, W, device, dtype=pred.dtype)
    # sums over B,H,W with mask; broadcast over channel
    abs_sum_c = (w * (pred - tgt).abs()).sum(dim=(0, 2, 3))  # (C,)
    sq_sum_c  = (w * (pred - tgt) ** 2).sum(dim=(0, 2, 3))   # (C,)
    pix_sum   = w.sum().item() * C / C  # scalar (per-channel same), use w.sum() directly for weighting
    return abs_sum_c.detach(), sq_sum_c.detach(), w.sum().detach()

def aggregate_final(abs_sum_c, sq_sum_c, w_pix_sum, band_weights=None):
    """
    abs_sum_c, sq_sum_c: torch tensors (C,), summed over entire dataset
    w_pix_sum: torch scalar = total valid pixels (summed over batches), but per-channel sum is same,
               so per-channel mean uses (w_pix_sum) and overall uses band-weights or equal weights.
    """
    C = abs_sum_c.numel()
    # Per-channel MAE/MSE (pixel-averaged)
    denom = w_pix_sum.clamp_min(1e-8)
    mae_c = abs_sum_c / denom
    mse_c = sq_sum_c / denom

    if band_weights is None:
        mae = mae_c.mean().item()
        mse = mse_c.mean().item()
    else:
        w = torch.tensor(band_weights, dtype=mae_c.dtype, device=mae_c.device)
        w = w / w.sum().clamp_min(1e-8)
        mae = (mae_c * w).sum().item()
        mse = (mse_c * w).sum().item()

    psnr = 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)
    # Per-channel PSNR
    psnr_c = torch.where(mse_c <= 1e-12, torch.full_like(mse_c, 99.0), 10.0 * torch.log10(1.0 / mse_c))
    return mae, mse, psnr, mae_c.cpu().numpy(), mse_c.cpu().numpy(), psnr_c.cpu().numpy()



# Visualization

def _stretch_to_uint8(x: torch.Tensor):
    x = x.detach().cpu().numpy()
    p2, p98 = np.percentile(x, [2, 98])
    if p98 - p2 < 1e-6: p98 = p2 + 1.0
    x = np.clip((x - p2) / (p98 - p2), 0, 1)
    return (x * 255).astype(np.uint8)

def _rgb_from_bands(t4: torch.Tensor):
    # target/pred order: [B2, B3, B4, B8]
    B2, B3, B4, B8 = t4[0], t4[1], t4[2], t4[3]
    rgb_true = np.dstack([_stretch_to_uint8(B4), _stretch_to_uint8(B3), _stretch_to_uint8(B2)])
    rgb_cir  = np.dstack([_stretch_to_uint8(B8), _stretch_to_uint8(B4), _stretch_to_uint8(B3)])
    return rgb_true, rgb_cir

def _hstack_compare(left_u8, right_u8, gap=6):
    h, w, _ = left_u8.shape
    canvas = np.ones((h, w * 2 + gap, 3), dtype=np.uint8) * 255
    canvas[:, :w] = left_u8
    canvas[:, w + gap:] = right_u8
    return canvas

def save_pred_gt(pre4, gt4, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    rgb_true_pred, rgb_cir_pred = _rgb_from_bands(pre4)
    rgb_true_gt,   rgb_cir_gt   = _rgb_from_bands(gt4)
    Image.fromarray(rgb_true_pred).save(os.path.join(out_dir, f"{stem}_pred_true.png"))
    Image.fromarray(rgb_true_gt  ).save(os.path.join(out_dir, f"{stem}_gt_true.png"))
    Image.fromarray(rgb_cir_pred ).save(os.path.join(out_dir, f"{stem}_pred_cir.png"))
    Image.fromarray(rgb_cir_gt   ).save(os.path.join(out_dir, f"{stem}_gt_cir.png"))
    cmp_true = _hstack_compare(rgb_true_pred, rgb_true_gt)
    cmp_cir  = _hstack_compare(rgb_cir_pred,  rgb_cir_gt)
    Image.fromarray(cmp_true).save(os.path.join(out_dir, f"{stem}_compare_true.png"))
    Image.fromarray(cmp_cir ).save(os.path.join(out_dir, f"{stem}_compare_cir.png"))



# Samplers

@torch.no_grad()
def model_eps(model, x_t, cond, t_idx):
    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
        return model(torch.cat([x_t, cond], dim=1), t_idx)

@torch.no_grad()
def ddpm_sample(model, cond, betas, alphas, alpha_bar, C_tgt):
    device = cond.device
    B, _, H, W = cond.shape
    x_t = torch.randn(B, C_tgt, H, W, device=device)
    for t in reversed(range(len(betas))):
        t_idx = torch.full((B,), t, device=device, dtype=torch.long)
        eps = model_eps(model, x_t, cond, t_idx)
        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bar[t]
        mean = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t + 1e-8)) * eps)
        if t > 0:
            x_t = mean + torch.sqrt(beta_t) * torch.randn_like(x_t)
        else:
            x_t = mean
    return torch.clamp(x_t, 0.0, 1.0)

@torch.no_grad()
def ddim_sample(model, cond, alphas, alpha_bar, C_tgt, steps=50):
    device = cond.device
    T = len(alphas)
    B, _, H, W = cond.shape
    x_t = torch.randn(B, C_tgt, H, W, device=device)

    # Robust index set: float → round → unique(sorted=True)
    idxs = torch.linspace(0, T - 1, steps, device=device)
    idxs = torch.round(idxs).to(torch.long)
    idxs = torch.unique(idxs, sorted=True)

    for i in reversed(range(len(idxs))):
        t = int(idxs[i].item())
        t_idx = torch.full((B,), t, device=device, dtype=torch.long)
        eps = model_eps(model, x_t, cond, t_idx)
        a_t = alpha_bar[t]
        x0 = (x_t - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t + 1e-8)
        if i == 0:
            x_t = x0
        else:
            a_prev = alpha_bar[int(idxs[i - 1].item())]
            x_t = torch.sqrt(a_prev) * x0 + torch.sqrt(1 - a_prev) * eps
    return torch.clamp(x_t, 0.0, 1.0)

@torch.no_grad()
def partial_ddim_from_gt(model, x_gt, cond, alpha_bar, k: int):
    """
    Forward to t=k with fresh noise, then reverse k→0 via DDIM (no extra noise).
    """
    device = x_gt.device
    B, C, H, W = x_gt.shape
    k = int(max(0, min(k, len(alpha_bar) - 1)))
    a_t = alpha_bar[k].view(1, 1, 1, 1)
    x_t = torch.sqrt(a_t) * x_gt + torch.sqrt(1 - a_t) * torch.randn_like(x_gt)

    idxs = torch.arange(k, -1, -1, device=device)
    for i in range(len(idxs) - 1):
        cur = int(idxs[i].item()); nxt = int(idxs[i + 1].item())
        t_idx = torch.full((B,), cur, device=device, dtype=torch.long)
        eps = model_eps(model, x_t, cond, t_idx)
        a_cur = alpha_bar[cur]; a_prev = alpha_bar[nxt]
        x0 = (x_t - torch.sqrt(1 - a_cur) * eps) / torch.sqrt(a_cur + 1e-8)
        x_t = torch.sqrt(a_prev) * x0 + torch.sqrt(1 - a_prev) * eps
    return torch.clamp(x_t, 0.0, 1.0)


# Main evaluation
def run_eval(
    patch_dir, ckpt_path, out_dir,
    mode="ddim", T=1000, time_schedule="cosine", ddim_steps=50, batch_size=2, base_ch=96,
    save_n=16, max_files=None, band_weights=None, partial_reverse_k=None, seed=0
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    # inspect channels
    npz_files = [f for f in os.listdir(patch_dir) if f.endswith(".npz")]
    assert len(npz_files) > 0, f"No .npz files found in {patch_dir}"
    _sample = np.load(os.path.join(patch_dir, npz_files[0]))
    C_cond = _sample["inputs"].shape[0]
    C_tgt  = _sample["target"].shape[0]
    print(f"[INFO] inputs={C_cond}, target={C_tgt}")

    # schedule
    betas  = make_schedule(T, time_schedule).to(device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    # data
    ds = S1toS2Dataset(patch_dir, max_files=max_files)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # model
    model = UNetSmall(in_ch=C_cond + C_tgt, out_ch=C_tgt, base_ch=base_ch).to(device)
    state = torch.load(ckpt_path, map_location=device)
    # tolerate various checkpoint formats
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()

    # accumulators (global pixel-weighted)
    abs_sum_c_total = torch.zeros(C_tgt, device=device)
    sq_sum_c_total  = torch.zeros(C_tgt, device=device)
    pix_sum_total   = torch.tensor(0.0, device=device)

    # For saving previews
    saved = 0
    pbar = tqdm(loader, desc=f"Inference [{mode}/{time_schedule}]")
    for i, (x_cond, x_gt, mask) in enumerate(pbar):
        x_cond = x_cond.to(device, non_blocking=True)
        x_gt   = x_gt.to(device, non_blocking=True)
        mask_b = mask.to(device, non_blocking=True) if mask is not None else None

        # sampling
        if mode == "ddpm":
            x_pred = ddpm_sample(model, x_cond, betas, alphas, alpha_bar, C_tgt)
        else:
            x_pred = ddim_sample(model, x_cond, alphas, alpha_bar, C_tgt, steps=ddim_steps)

        # accumulate pixel-weighted sums
        abs_sum_c, sq_sum_c, wsum = channelwise_error_sums(x_pred, x_gt, mask_b)
        abs_sum_c_total += abs_sum_c
        sq_sum_c_total  += sq_sum_c
        pix_sum_total   += wsum

        # quick status (batch-level rough MAE/PSNR for feel)
        with torch.no_grad():
            denom = wsum.clamp_min(1e-8)
            mae_c_b = abs_sum_c / denom
            mse_c_b = sq_sum_c  / denom
            mae_b = mae_c_b.mean().item()
            mse_b = mse_c_b.mean().item()
            psnr_b = (99.0 if mse_b <= 1e-12 else 10.0 * math.log10(1.0 / mse_b))
        pbar.set_postfix(mae=f"{mae_b:.4f}", psnr=f"{psnr_b:.2f}")

        # save a few visual comparisons
        bs = x_gt.size(0)
        for b in range(bs):
            if saved >= save_n: break
            pre4 = x_pred[b].detach().cpu()
            gt4  = x_gt[b].detach().cpu()
            stem = f"{mode}_{i:04d}_{b:02d}"
            save_pred_gt(pre4, gt4, out_dir, stem)
            np.save(os.path.join(out_dir, f"{stem}_pred.npy"), pre4.numpy())
            np.save(os.path.join(out_dir, f"{stem}_gt.npy"),   gt4.numpy())
            saved += 1

        # optional partial-reverse on the first batch only (diagnostic)
        if (partial_reverse_k is not None) and (i == 0):
            ks = [int(k) for k in partial_reverse_k]
            for k in ks:
                xr = partial_ddim_from_gt(model, x_gt, x_cond, alpha_bar, k)
                abs_k, sq_k, w_k = channelwise_error_sums(xr, x_gt, mask_b)
                denom = w_k.clamp_min(1e-8)
                mae_k = (abs_k / denom).mean().item()
                mse_k = (sq_k  / denom).mean().item()
                psnr_k = 99.0 if mse_k <= 1e-12 else 10.0 * math.log10(1.0 / mse_k)
                print(f"[partial-reverse k={k}] MAE={mae_k:.6f}  MSE={mse_k:.6f}  PSNR={psnr_k:.3f} dB")

    # final aggregation
    mae, mse, psnr, mae_c, mse_c, psnr_c = aggregate_final(
        abs_sum_c_total, sq_sum_c_total, pix_sum_total, band_weights=None
    )
    print("\n==== Unweighted (equal-channel) ====")
    print(f"MAE:  {mae:.6f}")
    print(f"MSE:  {mse:.6f}")
    print(f"PSNR: {psnr:.3f} dB")

    # weighted (if provided)
    if band_weights is not None:
        mae_w, mse_w, psnr_w, _, _, _ = aggregate_final(
            abs_sum_c_total, sq_sum_c_total, pix_sum_total, band_weights=band_weights
        )
        print("\n==== Weighted (band_weights) ====")
        print(f"band_weights = {band_weights}")
        print(f"MAE_w:  {mae_w:.6f}")
        print(f"MSE_w:  {mse_w:.6f}")
        print(f"PSNR_w: {psnr_w:.3f} dB")

    # per-channel report (pixel-weighted)
    names = ["B2", "B3", "B4", "B8"] if len(mae_c) == 4 else [f"Band{i}" for i in range(len(mae_c))]
    print("\n-- Per-channel metrics (pixel-weighted) --")
    for i, nm in enumerate(names):
        print(f"{nm:>3s}:  MAE={mae_c[i]:.6f}  MSE={mse_c[i]:.6f}  PSNR={psnr_c[i]:.3f} dB")

    print(f"\n[INFO] Results saved to: {out_dir}")



# CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_dir",   type=str, required=True)
    ap.add_argument("--ckpt",        type=str, required=True)
    ap.add_argument("--out_dir",     type=str, required=True)
    ap.add_argument("--mode",        type=str, default="ddim", choices=["ddpm", "ddim"])
    ap.add_argument("--T",           type=int, default=1000)
    ap.add_argument("--time_schedule", type=str, default="cosine", choices=["cosine", "linear"])
    ap.add_argument("--ddim_steps",  type=int, default=50)
    ap.add_argument("--batch_size",  type=int, default=2)
    ap.add_argument("--base_ch",     type=int, default=96)
    ap.add_argument("--save_n",      type=int, default=16)
    ap.add_argument("--max_files",   type=int, default=None, help="Limit number of patches for quick tests")
    ap.add_argument("--band_weights", nargs="*", type=float, default=None,
                    help="Per-target-channel weights for metrics, e.g., --band_weights 1 1 2 2")
    ap.add_argument("--partial_reverse_k", nargs="*", type=int, default=None,
                    help="e.g., --partial_reverse_k 50 100 200 (diagnostic)")
    ap.add_argument("--seed",        type=int, default=0)
    args = ap.parse_args()

    run_eval(
        patch_dir=args.patch_dir,
        ckpt_path=args.ckpt,
        out_dir=args.out_dir,
        mode=args.mode,
        T=args.T,
        time_schedule=args.time_schedule,
        ddim_steps=args.ddim_steps,
        batch_size=args.batch_size,
        base_ch=args.base_ch,
        save_n=args.save_n,
        max_files=args.max_files,
        band_weights=args.band_weights,
        partial_reverse_k=args.partial_reverse_k,
        seed=args.seed
    )
