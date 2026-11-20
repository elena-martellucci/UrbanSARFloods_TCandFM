"""Check that U-Net can train on a real UrbanSARFloods crop."""

import argparse
from time import perf_counter

import segmentation_models_pytorch as smp
import torch
from torch import nn

from dataset import IGNORE_LABEL, UrbanSARFloods
from temporal_features import add_all_differences, add_coherence_differences, add_intensity_differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/urban_sar_floods")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--differences", nargs="?", const="intensity", choices=("intensity", "coherence", "both"),
                        help="Append the VV and VH difference maps of intensity (default), coherence, or both")
    args = parser.parse_args()
    if args.steps < 1 or args.crop_size < 64 or args.crop_size % 32:
        parser.error("Use at least one step and a crop size divisible by 32, at least 64.")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("PyTorch cannot see a CUDA GPU in this environment.")

    torch.manual_seed(42)
    dataset = UrbanSARFloods(args.data_root, "train")
    image, mask = dataset[args.index]
    if args.differences:
        add_differences = {"intensity": add_intensity_differences, "coherence": add_coherence_differences,
                           "both": add_all_differences}[args.differences]
        image = add_differences(image)
    size = args.crop_size
    height, width = mask.shape
    if size > min(height, width):
        parser.error(f"Crop size {size} exceeds tile dimensions {mask.shape}.")
    row = (height - size) // 2
    col = (width - size) // 2
    x = torch.from_numpy(image[:, row:row + size, col:col + size].copy()).unsqueeze(0)
    y = torch.from_numpy(mask[row:row + size, col:col + size].copy()).unsqueeze(0)
    valid = y != IGNORE_LABEL
    if not valid.any():
        parser.error("The crop has no pixels with SAR data. Choose another tile with --index.")
    print("Crop pixel counts [non-flooded, flooded open, flooded urban]:",
          torch.bincount(y[valid], minlength=3).tolist(), flush=True)
    print("Crop pixels without SAR data, left out of the loss:", (~valid).sum().item(), flush=True)

    decibel_end = 8 if args.differences == "coherence" else 10
    x[:, 4:decibel_end] /= 20.0
    x = torch.nan_to_num(x)
    x, y = x.to(device), y.to(device)
    model = smp.Unet(encoder_name="resnet18", encoder_weights=None,
                     in_channels=x.shape[1], classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    print(f"Device: {device} | input: {tuple(x.shape)} | target: {tuple(y.shape)}", flush=True)

    step_times = []
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    for step in range(args.steps):
        start = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Non-finite loss at step {step + 1}")
        loss.backward()
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        step_times.append(perf_counter() - start)
        if step == 0 or step == args.steps - 1:
            print(f"Step {step + 1}: logits {tuple(logits.shape)}, loss {loss.item():.4f}", flush=True)

    if args.steps > 3:
        measured_times = step_times[3:]
        print(f"Mean training step after 3 warm-up steps: {sum(measured_times) / len(measured_times):.3f} s")
    else:
        print(f"Training step time including startup: {sum(step_times) / len(step_times):.3f} s")
    if device == "cuda":
        print(f"Peak allocated CUDA memory: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
