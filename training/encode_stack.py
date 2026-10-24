"""Apply a trained SimCLR or NNCLR encoder to an image stack and save the
512-d latents (used for Horizon-AGN and HSC).

    python encode_stack.py --ckpt <best.pt> --stack <images_X.npy> --out <latents_X.npy>

The stack must be preprocessed like images_tng.npy (128 px, arcsinh-stretched
to [0, 1]); see data/prepare_transfer_data.py.
"""
import argparse

import numpy as np
import torch

from common import ResNetEncoder


def load_encoder(ckpt_path, arm):
    """The arm is detected from the checkpoint keys unless given."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if arm == "auto":
        if "model" in ck:
            arm = "nnclr"
        elif "encoder" in ck:
            arm = "simclr"
        else:
            raise SystemExit(f"can't detect arm from ckpt keys {list(ck.keys())}; pass --arm")
    if arm == "nnclr":
        state = {k[len("encoder."):]: v for k, v in ck["model"].items()
                 if k.startswith("encoder.")}
    else:
        state = ck["encoder"]
    enc = ResNetEncoder()
    enc.load_state_dict(state)
    return enc, arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="auto", choices=["auto", "nnclr", "simclr"])
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc, arm = load_encoder(args.ckpt, args.arm)
    enc.eval().to(dev)
    print(f"loaded {arm} encoder from {args.ckpt}", flush=True)

    stk = np.load(args.stack, mmap_mode="r")
    N = len(stk)
    print(f"encoding {N:,} images from {args.stack} (shape {stk.shape})", flush=True)

    out = None
    with torch.no_grad():
        for i in range(0, N, args.batch):
            chunk = np.ascontiguousarray(stk[i:i + args.batch], dtype=np.float32)
            x = torch.from_numpy(chunk).unsqueeze(1).to(dev, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
                z = enc(x)
            z = z.float().cpu().numpy()
            if out is None:
                out = np.empty((N, z.shape[1]), dtype=np.float32)
                print(f"  latent dim {z.shape[1]}", flush=True)
            out[i:i + len(z)] = z
            if i % (args.batch * 50) == 0:
                print(f"  {i:,}/{N:,}", flush=True)
    np.save(args.out, out)
    print(f"saved latents {out.shape} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
