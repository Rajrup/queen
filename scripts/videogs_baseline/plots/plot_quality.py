"""
Plot GT vs decompressed model quality per frame (PSNR and SSIM).
Requires: evaluation_results.csv under input_folder/<output_tag>/evaluation/
"""
import os
import argparse
import csv
import matplotlib.pyplot as plt

def build_output_tag(args):
    return f"qp_{args.qp}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Base folder for videogs_compression")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--sequence_name", type=str, required=True)
    parser.add_argument("--qp", type=int, required=True)
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder for plot PNG")
    args = parser.parse_args()

    output_tag = build_output_tag(args)
    config_dir = os.path.join(args.input_folder, output_tag)
    evaluation_csv = os.path.join(config_dir, "evaluation", "evaluation_results.csv")
    out_dir = os.path.join(args.output_folder, "plots", args.dataset_name, args.sequence_name, output_tag)
    out_path = os.path.join(out_dir, "quality.png")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(evaluation_csv):
        raise SystemExit(f"Required file not found: {evaluation_csv}")

    frames, gt_psnr, decomp_psnr, gt_ssim, decomp_ssim = [], [], [], [], []
    with open(evaluation_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            if row["frame_id"] == "avg":
                continue
            frames.append(int(row["frame_id"]))
            gt_psnr.append(float(row["gt_psnr"]))
            decomp_psnr.append(float(row["decomp_psnr"]))
            gt_ssim.append(float(row["gt_ssim"]))
            decomp_ssim.append(float(row["decomp_ssim"]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    n = len(frames)
    x = range(n)
    tick_every = 10

    # --- PSNR ---
    ax1.plot(x, gt_psnr, "o-", label="GT model", color="green", markersize=4)
    ax1.plot(x, decomp_psnr, "s-", label="Decompressed model", color="coral", markersize=4)
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title(f"VideoGS quality per frame: GT vs Decompressed [{output_tag}]\n{args.dataset_name}/{args.sequence_name}")
    avg_gt_p = sum(gt_psnr) / n
    avg_dec_p = sum(decomp_psnr) / n
    ax1.axhline(y=avg_gt_p, color="green", linestyle="--", alpha=0.4,
                label=f"GT avg = {avg_gt_p:.2f}")
    ax1.axhline(y=avg_dec_p, color="coral", linestyle="--", alpha=0.4,
                label=f"Decomp avg = {avg_dec_p:.2f}")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- SSIM ---
    ax2.plot(x, gt_ssim, "o-", label="GT model", color="green", markersize=4)
    ax2.plot(x, decomp_ssim, "s-", label="Decompressed model", color="coral", markersize=4)
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("SSIM")
    avg_gt_s = sum(gt_ssim) / n
    avg_dec_s = sum(decomp_ssim) / n
    ax2.axhline(y=avg_gt_s, color="green", linestyle="--", alpha=0.4,
                label=f"GT avg = {avg_gt_s:.4f}")
    ax2.axhline(y=avg_dec_s, color="coral", linestyle="--", alpha=0.4,
                label=f"Decomp avg = {avg_dec_s:.4f}")
    ax2.set_xticks(list(x)[::tick_every])
    ax2.set_xticklabels(frames[::tick_every], rotation=90)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
