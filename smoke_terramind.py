"""Run the TerraMind Tiny forward and segmentation-head smoke tests."""

import argparse
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F
from terratorch import BACKBONE_REGISTRY

from dataset import UrbanSARFloods


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("forward", "head-step"))
    parser.add_argument("--data-root", default="data/smoke_flooded")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("This smoke test requires a CUDA GPU.")

    torch.manual_seed(42)
    image, mask = UrbanSARFloods(args.data_root, "train")[args.index]
    size = 256
    height, width = mask.shape
    if min(height, width) < size:
        parser.error("The selected tile is smaller than the 256 by 256 crop.")
    row = (height - size) // 2
    col = (width - size) // 2
    rows = slice(row, row + size)
    cols = slice(col, col + size)

    # GeoTIFF bands 8 and 7 are the post-event VV and VH intensities in dB.
    # Use the S1GRD statistics from IBM's TerraMind flood example.
    inputs = torch.from_numpy(image[[7, 6], rows, cols]).unsqueeze(0).cuda()
    mean = inputs.new_tensor([-12.599, -20.293]).view(1, 2, 1, 1)
    std = inputs.new_tensor([5.195, 5.890]).view(1, 2, 1, 1)
    inputs = (inputs - mean) / std
    target = torch.from_numpy(mask[rows, cols].copy()).unsqueeze(0).cuda()

    print("Input [batch, channels, height, width]:", tuple(inputs.shape), flush=True)
    print("Urban-flood pixels:", (target == 2).sum().item(), flush=True)
    print("Loading pretrained TerraMind Tiny...", flush=True)
    encoder = BACKBONE_REGISTRY.build(
        "terramind_v1_tiny", pretrained=True, modalities=["S1GRD"]
    ).eval().requires_grad_(False).cuda()

    if args.mode == "forward":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        with torch.inference_mode():
            features = encoder({"S1GRD": inputs})
        torch.cuda.synchronize()
        elapsed = perf_counter() - start
        if len(features) == 0 or not all(torch.isfinite(layer).all().item() for layer in features):
            raise RuntimeError("Expected finite feature tensors from TerraMind Tiny.")
        print("Feature shapes:", [tuple(layer.shape) for layer in features])
        print(f"First forward pass: {elapsed:.3f} s")
    else:
        head = nn.Conv2d(192, 3, kernel_size=1).cuda()
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
        weight_before = head.weight.detach().clone()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            tokens = encoder({"S1GRD": inputs})[-1]
        if tokens.shape != (1, 256, 192):
            raise RuntimeError(f"Unexpected last-layer shape: {tuple(tokens.shape)}")

        grid = tokens.transpose(1, 2).reshape(1, 192, 16, 16)
        logits = F.interpolate(
            head(grid), size=target.shape[-2:], mode="bilinear", align_corners=False
        )
        loss = F.cross_entropy(logits, target)
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite segmentation loss.")
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = perf_counter() - start
        if torch.equal(weight_before, head.weight):
            raise RuntimeError("Segmentation head weights did not update.")

        print("Logits:", tuple(logits.shape))
        print(f"Step 1 loss: {loss.item():.4f}")
        print("Head weights updated: yes")
        print(f"Training step including startup: {elapsed:.3f} s")

    print(f"Peak allocated CUDA memory: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
