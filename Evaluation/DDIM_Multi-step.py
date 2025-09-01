import os, math, argparse, csv
from typing import Tuple, List, Optional, Dict
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

# schedule 
def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = T + 1
    t = torch.linspace(0, T, steps, dtype=torch.float64)
    f = torch.cos(((t / T + s) / (1 + s)) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    betas = torch.clip(betas, 1e-5, 0.999)
    return betas.float()

# model 
class UNetSmall(nn.Module):
    """UNet used in training; time index injected as a scalar map channel."""
    def __init__(self, in_ch: int, out_ch: int, base_ch: int = 96):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=False),
                nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=False),
            )
        self.inc   = nn.Sequential(nn.Conv2d(in_ch + 1, base_ch, 3, padding=1), nn.ReLU(inplace=False))
        self.down1 = nn.Sequential(block(base_ch,   base_ch*2), nn.MaxPool2d(2))
        self.down2 = nn.Sequential(block(base_ch*2, base_ch*4), nn.MaxPool2d(2))
        self.down3 = nn.Sequential(block(base_ch*4, base_ch*8), nn.MaxPool2d(2))

        self.up3   = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.conv3 = block(base_ch*8, base_ch*4)
        self.up2   = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.conv2 = block(base_ch*4, base_ch*2)
        self.up1   = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.conv1 = block(base_ch*2, base_ch)

        self.outc  = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, xt_and_cond: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        B, _, H, W = xt_and_cond.shape
        t_map = t_idx.view(B,1,1,1).float().repeat(1,1,H,W)
        x = torch.cat([xt_and_cond, t_map], dim=1)
        e1 = self.inc(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        d3 = self.up3(e4); d3 = torch.cat([d3,e3], dim=1); d3 = self.conv3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2,e2], dim=1); d2 = self.conv2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1,e1], dim=1); d1 = self.conv1(d1)
        return self.outc(d1)

# viz helpers 
def _stretch_to_uint8(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().numpy()
    p2, p98 = np.percentile(x, [2, 98])
    if p98 - p2 < 1e-6: p98 = p2 + 1.0
    x = np.clip((x - p2) / (p98 - p2), 0, 1)
    return (x * 255).astype(np.uint8)

def save_rgb_triplet(t4: torch.Tensor, path_true: str, path_cir: str):
    """Assumes [B2,B3,B4,B8] for quick True/CIR previews."""
    B2,B3,B4,B8 = t4[0], t4[1], t4[2], t4[3]
    true_rgb = np.dstack([_stretch_to_uint8(B4), _stretch_to_uint8(B3), _stretch_to_uint8(B2)])
    cir_rgb  = np.dstack([_stretch_to_uint8(B8), _stretch_to_uint8(B4), _stretch_to_uint8(B3)])
    Image.fromarray(true_rgb).save(path_true)
    Image.fromarray(cir_rgb).save(path_cir)

# metrics 
def masked_mae(pred, tgt, mask=None) -> float:
    if mask is None:
        w = torch.ones_like(pred[:, :1])
    else:
        w = (mask.unsqueeze(1) if mask.ndim==3 else mask).float().to(pred.device)
        w = (w > 0).float()
    num = (w * (pred - tgt).abs()).sum()
    den = w.sum() * pred.size(1)
    return (num / (den + 1e-8)).item()

def masked_mse(pred, tgt, mask=None) -> float:
    if mask is None:
        w = torch.ones_like(pred[:, :1])
    else:
        w = (mask.unsqueeze(1) if mask.ndim==3 else mask).float().to(pred.device)
        w = (w > 0).float()
    num = (w * (pred - tgt)**2).sum()
    den = w.sum() * pred.size(1)
    return (num / (den + 1e-8)).item()

def psnr(pred, tgt, mask=None) -> float:
    m = masked_mse(pred, tgt, mask)
    if m <= 1e-12: return 99.0
    return 10.0 * math.log10(1.0 / m)

def ssim_simple(pred, tgt, C1=0.01**2, C2=0.03**2) -> float:
    mu_x = pred.mean().item(); mu_y = tgt.mean().item()
    vx = pred.var().item();    vy = tgt.var().item()
    cxy = ((pred - pred.mean()) * (tgt - tgt.mean())).mean().item()
    return ((2*mu_x*mu_y + C1) * (2*cxy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (vx + vy + C2) + 1e-8)

# I/O 
def load_npz_as_tensors(path: str, device: torch.device):
    d = np.load(path)
    x_cond = torch.from_numpy(np.nan_to_num(d["inputs"].astype(np.float32))).unsqueeze(0).to(device)
    x_gt   = torch.from_numpy(np.nan_to_num(d["target"].astype(np.float32))).unsqueeze(0).to(device)
    mask   = None
    if "mask" in d:
        mask = torch.from_numpy(np.nan_to_num(d["mask"].astype(np.float32))).unsqueeze(0).to(device)
    return x_cond, x_gt, mask, x_cond.size(1), x_gt.size(1)

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

# core evals 
def ddim_multistep_eval(model, x_gt, x_cond, alpha_bar, mask, t_start=200, steps=20):
    """Deterministic DDIM (eta=0) from t_start -> 0 with K=steps."""
    t_start = max(1, min(int(t_start), len(alpha_bar)-1))
    dev = x_gt.device
    t_idx = torch.full((1,), t_start, dtype=torch.long, device=dev)
    a_t = alpha_bar[t_idx].view(-1,1,1,1)
    noise = torch.randn_like(x_gt)
    x_t = torch.sqrt(a_t)*x_gt + torch.sqrt(1 - a_t)*noise
    ts = torch.linspace(t_start, 0, steps+1, dtype=torch.long, device=dev)
    with torch.no_grad():
        for i in range(steps):
            t_cur  = ts[i].view(1)
            t_next = ts[i+1].view(1)
            a_cur  = alpha_bar[t_cur].view(-1,1,1,1)
            a_next = alpha_bar[t_next].view(-1,1,1,1)
            eps = model(torch.cat([x_t, x_cond], dim=1), t_cur)
            x0_hat = (x_t - torch.sqrt(1 - a_cur)*eps) / torch.sqrt(a_cur + 1e-8)
            x_t = torch.sqrt(a_next)*x0_hat + torch.sqrt(1 - a_next)*eps
        x0_hat_final = torch.clamp(x0_hat, 0.0, 1.0)
    mae = masked_mae(x0_hat_final, x_gt, mask)
    mse = masked_mse(x0_hat_final, x_gt, mask)
    return mae, mse, x0_hat_final

def eps_diagnostics(model, x_gt, x_cond, alpha_bar, t_small: int):
    """Return (eps_MSE, cosine(pred_eps, z))."""
    dev = x_gt.device
    t_small = max(1, min(int(t_small), len(alpha_bar)-1))
    t_idx = torch.full((1,), t_small, dtype=torch.long, device=dev)
    a_t = alpha_bar[t_idx].view(-1,1,1,1)
    z   = torch.randn_like(x_gt)
    x_t = torch.sqrt(a_t)*x_gt + torch.sqrt(1 - a_t)*z
    with torch.no_grad():
        pred_eps = model(torch.cat([x_t, x_cond], dim=1), t_idx)
    mse = torch.mean((pred_eps - z)**2).item()
    cos = torch.sum(pred_eps*z).item() / (
        torch.sqrt(torch.sum(pred_eps**2)).item() * torch.sqrt(torch.sum(z**2)).item() + 1e-8
    )
    return mse, cos

def one_step_recon(model, x_gt, x_cond, alpha_bar, mask, t_small: int, rng_seed: Optional[int] = None):
    """One-step reconstruction (for seed_stats baseline)."""
    if rng_seed is not None: torch.manual_seed(rng_seed)
    dev = x_gt.device
    t_small = max(1, min(int(t_small), len(alpha_bar)-1))
    t_idx = torch.full((1,), t_small, dtype=torch.long, device=dev)
    a_t = alpha_bar[t_idx].view(-1,1,1,1)
    noise = torch.randn_like(x_gt)
    x_t = torch.sqrt(a_t)*x_gt + torch.sqrt(1 - a_t)*noise
    with torch.no_grad():
        eps = model(torch.cat([x_t, x_cond], dim=1), t_idx)
        x0_hat = (x_t - torch.sqrt(1 - a_t)*eps) / torch.sqrt(a_t + 1e-8)
        x0_hat = torch.clamp(x0_hat, 0.0, 1.0)
    mae = masked_mae(x0_hat, x_gt, mask)
    mse = masked_mse(x0_hat, x_gt, mask)
    return mae, mse, x0_hat

# main (batch) 
def main():
    ap = argparse.ArgumentParser("Batch eval suite")
    ap.add_argument("--mode", required=True, choices=["ddim","eps","seed_stats","per_band","ablate"])
    ap.add_argument("--patch_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--base_ch", type=int, default=96)
    # batch control
    ap.add_argument("--max_files", type=int, default=0, help="0 means ALL .npz in patch_dir")
    ap.add_argument("--save_viz_n", type=int, default=6, help="save previews for first N samples")
    # DDIM
    ap.add_argument("--t_start", type=int, default=200)
    ap.add_argument("--ddim_steps", type=int, default=20)
    # eps/seed_stats/per_band/ablate
    ap.add_argument("--t_small", type=int, default=20)
    ap.add_argument("--n_seeds", type=int, default=8)
    ap.add_argument("--seed_base", type=int, default=1234)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # File list
    files = sorted([f for f in os.listdir(args.patch_dir) if f.endswith(".npz")])
    assert files, "No .npz found"
    if args.max_files > 0:
        files = files[:args.max_files]
    print(f"[INFO] Evaluating {len(files)} files")

    # Warm up first file to build the model
    x_cond0, x_gt0, mask0, Cc0, Ct0 = load_npz_as_tensors(os.path.join(args.patch_dir, files[0]), device)
    model = UNetSmall(in_ch=Cc0 + Ct0, out_ch=Ct0, base_ch=args.base_ch).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    betas = cosine_beta_schedule(args.T).to(device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    # Common preview dir
    viz_dir = os.path.join(args.out_dir, "previews"); ensure_dir(viz_dir)

    # MODE: DDIM 
    if args.mode == "ddim":
        csv_path = os.path.join(args.out_dir, "ddim_metrics.csv")
        maes, mses = [], []
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["file","t_start","ddim_steps","MAE","MSE"])
            for i, fname in enumerate(files):
                x_cond, x_gt, mask, Cc, Ct = load_npz_as_tensors(os.path.join(args.patch_dir, fname), device)
                assert Cc==Cc0 and Ct==Ct0, "Channel mismatch across files"
                mae, mse, x0 = ddim_multistep_eval(model, x_gt, x_cond, alpha_bar, mask,
                                                   t_start=args.t_start, steps=args.ddim_steps)
                maes.append(mae); mses.append(mse)
                w.writerow([fname, args.t_start, args.ddim_steps, f"{mae:.6f}", f"{mse:.6f}"])
                if i < args.save_viz_n:
                    save_rgb_triplet(x0[0].cpu(),
                                     os.path.join(viz_dir, f"{i:03d}_ddim_pred_true.png"),
                                     os.path.join(viz_dir, f"{i:03d}_ddim_pred_cir.png"))
            # summary
        maes_t = torch.tensor(maes); mses_t = torch.tensor(mses)
        with open(os.path.join(args.out_dir, "ddim_summary.txt"), "w") as f:
            f.write(f"files: {len(files)}  t_start: {args.t_start}  steps: {args.ddim_steps}\n")
            f.write(f"MAE mean/std: {maes_t.mean().item():.6f} / {maes_t.std(unbiased=False).item():.6f}\n")
            f.write(f"MSE mean/std: {mses_t.mean().item():.6f} / {mses_t.std(unbiased=False).item():.6f}\n")
        print("[DONE] DDIM")

    # MODE: EPS 
    elif args.mode == "eps":
        csv_path = os.path.join(args.out_dir, "eps_diag.csv")
        mses, coses = [], []
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["file","t_small","eps_MSE","cosine"])
            for fname in files:
                x_cond, x_gt, mask, Cc, Ct = load_npz_as_tensors(os.path.join(args.patch_dir, fname), device)
                assert Cc==Cc0 and Ct==Ct0, "Channel mismatch across files"
                mse, cos = eps_diagnostics(model, x_gt, x_cond, alpha_bar, t_small=args.t_small)
                mses.append(mse); coses.append(cos)
                w.writerow([fname, args.t_small, f"{mse:.6f}", f"{cos:.6f}"])
        mses_t = torch.tensor(mses); coses_t = torch.tensor(coses)
        with open(os.path.join(args.out_dir, "eps_summary.txt"), "w") as f:
            f.write(f"files: {len(files)}  t_small: {args.t_small}\n")
            f.write(f"eps_MSE mean/std: {mses_t.mean().item():.6f} / {mses_t.std(unbiased=False).item():.6f}\n")
            f.write(f"cosine  mean/std: {coses_t.mean().item():.6f} / {coses_t.std(unbiased=False).item():.6f}\n")
        print("[DONE] EPS")

    # MODE: SEED_STATS 
    elif args.mode == "seed_stats":
        csv_path = os.path.join(args.out_dir, "seed_stats.csv")
        # Collect per-file mean/std, then compute dataset mean of means (and of stds)
        mae_means, mae_stds, mse_means, mse_stds = [], [], [], []
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["file","t_small","n_seeds","MAE_mean","MAE_std","MSE_mean","MSE_std"])
            for fname in files:
                x_cond, x_gt, mask, Cc, Ct = load_npz_as_tensors(os.path.join(args.patch_dir, fname), device)
                assert Cc==Cc0 and Ct==Ct0, "Channel mismatch across files"
                maes, mses = [], []
                for s in range(args.n_seeds):
                    mae, mse, _ = one_step_recon(model, x_gt, x_cond, alpha_bar, mask,
                                                 t_small=args.t_small, rng_seed=args.seed_base + s)
                    maes.append(mae); mses.append(mse)
                maes_t = torch.tensor(maes); mses_t = torch.tensor(mses)
                mae_mu, mae_sd = maes_t.mean().item(), maes_t.std(unbiased=False).item()
                mse_mu, mse_sd = mses_t.mean().item(), mses_t.std(unbiased=False).item()
                mae_means.append(mae_mu); mae_stds.append(mae_sd)
                mse_means.append(mse_mu); mse_stds.append(mse_sd)
                w.writerow([fname, args.t_small, args.n_seeds,
                            f"{mae_mu:.6f}", f"{mae_sd:.6f}", f"{mse_mu:.6f}", f"{mse_sd:.6f}"])
        with open(os.path.join(args.out_dir, "seed_stats_summary.txt"), "w") as f:
            f.write(f"files: {len(files)}  t_small: {args.t_small}  n_seeds: {args.n_seeds}\n")
            f.write(f"MAE mean_of_means/std_of_means: "
                    f"{np.mean(mae_means):.6f} / {np.std(mae_means):.6f}\n")
            f.write(f"MSE mean_of_means/std_of_means: "
                    f"{np.mean(mse_means):.6f} / {np.std(mse_means):.6f}\n")
            f.write(f"Avg per-file MAE_std: {np.mean(mae_stds):.6f}\n")
            f.write(f"Avg per-file MSE_std: {np.mean(mse_stds):.6f}\n")
        print("[DONE] SEED_STATS")

    # MODE: PER_BAND 
    elif args.mode == "per_band":
        # Using one-step reconstruction at t_small for each file (fresh noise)
        # Aggregate per-band metrics across dataset.
        band_accum: Dict[int, List[float]] = {}  # band -> list of metric tuples
        csv_path = os.path.join(args.out_dir, "per_band_all.csv")
        wrote_header = False
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            for i, fname in enumerate(files):
                x_cond, x_gt, mask, Cc, Ct = load_npz_as_tensors(os.path.join(args.patch_dir, fname), device)
                assert Cc==Cc0 and Ct==Ct0, "Channel mismatch across files"
                # one-step recon
                mae, mse, x0 = one_step_recon(model, x_gt, x_cond, alpha_bar, mask, t_small=args.t_small)
                # per-band stats
                C = x_gt.size(1)
                row = ["file", fname]
                if not wrote_header:
                    hdr = ["file"]
                    for b in range(C):
                        hdr += [f"band{b}_MAE", f"band{b}_MSE", f"band{b}_PSNR", f"band{b}_SSIMs"]
                    w.writerow(hdr); wrote_header = True
                per_band_rows = []
                for b in range(C):
                    p = x0[:,b:b+1]; g = x_gt[:,b:b+1]
                    mae_b = masked_mae(p,g,mask); mse_b = masked_mse(p,g,mask)
                    psnr_b = psnr(p,g,mask); ssim_b = ssim_simple(p,g)
                    per_band_rows += [f"{mae_b:.6f}", f"{mse_b:.6f}", f"{psnr_b:.3f}", f"{ssim_b:.4f}"]
                    if b not in band_accum: band_accum[b] = []
                    band_accum[b].append((mae_b, mse_b, psnr_b, ssim_b))
                w.writerow([fname] + per_band_rows)
                if i < args.save_viz_n:
                    save_rgb_triplet(x0[0].cpu(),
                                     os.path.join(viz_dir, f"{i:03d}_pb_pred_true.png"),
                                     os.path.join(viz_dir, f"{i:03d}_pb_pred_cir.png"))
        # aggregate per band
        agg_csv = os.path.join(args.out_dir, "per_band_summary.csv")
        with open(agg_csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["band","MAE_mean","MAE_std","MSE_mean","MSE_std","PSNR_mean","PSNR_std","SSIMs_mean","SSIMs_std"])
            for b, vals in sorted(band_accum.items()):
                arr = np.array(vals)  # N x 4
                mae_mu, mae_sd = arr[:,0].mean(), arr[:,0].std()
                mse_mu, mse_sd = arr[:,1].mean(), arr[:,1].std()
                psn_mu, psn_sd = arr[:,2].mean(), arr[:,2].std()
                ssi_mu, ssi_sd = arr[:,3].mean(), arr[:,3].std()
                w.writerow([b, f"{mae_mu:.6f}", f"{mae_sd:.6f}", f"{mse_mu:.6f}", f"{mse_sd:.6f}",
                            f"{psn_mu:.3f}", f"{psn_sd:.3f}", f"{ssi_mu:.4f}", f"{ssi_sd:.4f}"])
        print("[DONE] PER_BAND")

    # MODE: ABLATE 
    elif args.mode == "ablate":
        # For each cond channel, compute MAE/MSE with that channel zeroed; aggregate over dataset.
        # Also report deltas vs. non-ablated baseline using SAME noise per file.
        csv_path = os.path.join(args.out_dir, "ablate_all.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["file","t_small","baseline_MAE","baseline_MSE","channel","MAE","MSE","dMAE","dMSE"])
            # Prepare accumulators per channel
            ch_stats: Dict[int, List[Tuple[float,float,float,float]]] = {}  # ch -> list of (MAE, MSE, dMAE, dMSE)
            for fname in files:
                x_cond, x_gt, mask, Cc, Ct = load_npz_as_tensors(os.path.join(args.patch_dir, fname), device)
                assert Cc==Cc0 and Ct==Ct0, "Channel mismatch across files"
                # Build baseline with fixed noise for comparability within this file
                t_small = max(1, min(args.t_small, args.T-1))
                t_idx = torch.full((1,), t_small, dtype=torch.long, device=device)
                a_t = alpha_bar[t_idx].view(-1,1,1,1)
                torch.manual_seed(args.seed_base)
                base_noise = torch.randn_like(x_gt)
                x_t_base = torch.sqrt(a_t)*x_gt + torch.sqrt(1 - a_t)*base_noise
                with torch.no_grad():
                    eps_b = model(torch.cat([x_t_base, x_cond], dim=1), t_idx)
                    x0_b = (x_t_base - torch.sqrt(1 - a_t)*eps_b) / torch.sqrt(a_t + 1e-8)
                    x0_b = torch.clamp(x0_b, 0.0, 1.0)
                base_mae = masked_mae(x0_b, x_gt, mask); base_mse = masked_mse(x0_b, x_gt, mask)

                for ch in range(Cc):
                    # reuse base_noise to keep noise identical across ablations for this file
                    x_t = torch.sqrt(a_t)*x_gt + torch.sqrt(1 - a_t)*base_noise
                    x_cond_ab = x_cond.clone(); x_cond_ab[:, ch:ch+1].zero_()
                    with torch.no_grad():
                        eps = model(torch.cat([x_t, x_cond_ab], dim=1), t_idx)
                        x0_hat = (x_t - torch.sqrt(1 - a_t)*eps) / torch.sqrt(a_t + 1e-8)
                        x0_hat = torch.clamp(x0_hat, 0.0, 1.0)
                    mae = masked_mae(x0_hat, x_gt, mask); mse = masked_mse(x0_hat, x_gt, mask)
                    dMAE, dMSE = mae - base_mae, mse - base_mse
                    w.writerow([fname, t_small, f"{base_mae:.6f}", f"{base_mse:.6f}", ch,
                                f"{mae:.6f}", f"{mse:.6f}", f"{dMAE:.6f}", f"{dMSE:.6f}"])
                    ch_stats.setdefault(ch, []).append((mae, mse, dMAE, dMSE))
        # Aggregate across dataset per channel
        agg_path = os.path.join(args.out_dir, "ablate_summary.csv")
        with open(agg_path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["channel","MAE_mean","MAE_std","MSE_mean","MSE_std","dMAE_mean","dMAE_std","dMSE_mean","dMSE_std"])
            for ch, vals in sorted(ch_stats.items()):
                arr = np.array(vals)  # N x 4
                w.writerow([
                    ch,
                    f"{arr[:,0].mean():.6f}", f"{arr[:,0].std():.6f}",
                    f"{arr[:,1].mean():.6f}", f"{arr[:,1].std():.6f}",
                    f"{arr[:,2].mean():.6f}", f"{arr[:,2].std():.6f}",
                    f"{arr[:,3].mean():.6f}", f"{arr[:,3].std():.6f}",
                ])
        print("[DONE] ABLATE")
    

    else:
        raise ValueError("Unknown mode")

if __name__ == "__main__":
    main()
