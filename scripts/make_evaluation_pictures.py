#!/usr/bin/env python
"""Generate presentation-ready evaluation pictures.

Creates a folder with composite images showing:
- Original (input) vs generated (output)
- CTR for both
- Classifier prediction + confidence for both

Typical usage (presentation defaults):
    python scripts/make_evaluation_pictures.py \
        --input-dir data/raw/cardiomegaly \
        --checkpoint models/inpainting/final_model \
        --best-num 4 --worst-num 4 --random-num 4 \
        --output-dir Evaluation_pictures \
        --save-comparison
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path


def _try_load_temperature_from_calibration_json(path: Path) -> Optional[float]:
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        t = data.get("temperature", None)
        if t is None:
            return None
        t_f = float(t)
        if not (t_f > 0.0):
            return None
        return t_f
    except Exception:
        return None


def _format_ctr(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _format_pred(pred: Optional[str], conf: Optional[float]) -> str:
    if pred is None:
        return "n/a"
    if conf is None:
        return f"{pred}"
    return f"{pred} ({conf:.2f})"


def _format_ssim(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _load_default_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", 18)
    except Exception:
        return ImageFont.load_default()


def _normalize_single_xray(img: Image.Image) -> Image.Image:
    """Best-effort normalization to a single chest X-ray image.

    If the input is accidentally a side-by-side comparison (very wide aspect
    ratio), crop to the left half.
    """
    if img is None:
        return img
    w, h = img.size
    if h > 0 and (w / h) >= 1.6:
        # Likely a side-by-side comparison image. Keep left half.
        img = img.crop((0, 0, w // 2, h))
    return img


def _to_rgb_512(img: Image.Image, size: int = 512) -> Image.Image:
    img = _normalize_single_xray(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    return img


def _make_panel(
    original: Image.Image,
    generated: Image.Image,
    input_ctr: Optional[float],
    output_ctr: Optional[float],
    input_pred: Optional[str],
    input_conf: Optional[float],
    output_pred: Optional[str],
    output_conf: Optional[float],
    ssim_outside: Optional[float],
    title: str,
    classifier_temperature: Optional[float] = None,
    classifier_calibrated: bool = False,
) -> Image.Image:
    """Create a single composite image with text overlays."""

    font = _load_default_font()

    orig = _to_rgb_512(original)
    gen = _to_rgb_512(generated)

    pad = 12
    header_h = 104

    w = orig.width + gen.width + pad * 3
    h = orig.height + header_h + pad * 2

    canvas = Image.new("RGB", (w, h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    # Header text
    x0 = pad
    y0 = pad

    draw.text((x0, y0), title, fill=(255, 255, 255), font=font)
    draw.text(
        (x0, y0 + 24),
        f"Input  | CTR: {_format_ctr(input_ctr)} | Classifier score*: {_format_pred(input_pred, input_conf)}",
        fill=(220, 220, 220),
        font=font,
    )
    draw.text(
        (x0, y0 + 48),
        f"Output | CTR: {_format_ctr(output_ctr)} | Classifier score*: {_format_pred(output_pred, output_conf)}",
        fill=(220, 220, 220),
        font=font,
    )
    draw.text(
        (x0, y0 + 72),
        f"SSIM (outside mask / preserved anatomy): {_format_ssim(ssim_outside)}",
        fill=(220, 220, 220),
        font=font,
    )

    # Footnote-style clarification to avoid over-interpreting classifier output.
    temp_txt = f" (T={classifier_temperature:g})" if classifier_temperature is not None else ""
    note = "*softmax score, temperature-scaled" if classifier_calibrated else "*softmax score, not calibrated"
    draw.text(
        (w - pad - 380, y0 + 80),
        f"{note}{temp_txt}",
        fill=(180, 180, 180),
        font=font,
    )

    # Images
    img_y = pad + header_h
    canvas.paste(orig, (pad, img_y))
    canvas.paste(gen, (pad * 2 + orig.width, img_y))

    # Divider
    div_x = pad + orig.width + pad // 2
    draw.line([(div_x, img_y), (div_x, img_y + orig.height)], fill=(120, 120, 120), width=2)

    return canvas


def _compute_ctr(segmenter, validator, image: Image.Image) -> Optional[float]:
    try:
        masks = segmenter.segment(image)
        return validator.calculate_ctr(image, heart_mask=masks["heart"], lung_mask=masks["lungs"])
    except Exception:
        return None


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _healthy_probability(pred: Optional[str], conf: Optional[float]) -> float:
    """Return P(healthy) from (pred, conf).

    Assumes binary classifier where conf is the probability of the predicted class.
    """
    if pred is None or conf is None:
        return 0.0
    pred_l = str(pred).lower()
    if pred_l == "healthy":
        return float(_clamp01(float(conf)))
    # For binary case, approximate P(healthy) as 1 - P(cardiomegaly)
    return float(_clamp01(1.0 - float(conf)))


def _ctr_score(out_ctr: Optional[float], min_ctr: float, max_ctr: float) -> float:
    """Score CTR with a "healthy threshold" interpretation.

    In this project context, CTR below the upper threshold is generally acceptable.
    So we treat all CTR <= max_ctr as equally good (score=1.0) and only penalize
    values above max_ctr.

    The min_ctr parameter is kept for compatibility but not used for penalizing
    low CTR.
    """
    if out_ctr is None:
        return 0.0
    ctr = float(out_ctr)
    if ctr <= max_ctr:
        return 1.0
    # Soft margin beyond the upper bound.
    margin = 0.10
    return float(_clamp01(1.0 - (ctr - max_ctr) / margin))


def _composite_score(
    out_pred: Optional[str],
    out_conf: Optional[float],
    ssim_outside: Optional[float],
    out_ctr: Optional[float],
    min_ctr: float,
    max_ctr: float,
) -> float:
    """Single number for ranking best/worst examples.

    Higher is better.
    - Prefer output predicted healthy with high confidence
    - Prefer high SSIM outside mask (preserved anatomy)
    - Prefer CTR at/under the threshold (penalize only above max)
    """
    hp = _healthy_probability(out_pred, out_conf)
    ssim = float(ssim_outside) if ssim_outside is not None else 0.0
    ssim = _clamp01(ssim)
    ctr_s = _ctr_score(out_ctr, min_ctr=min_ctr, max_ctr=max_ctr)
    return float(0.40 * hp + 0.40 * ssim + 0.20 * ctr_s)


def _ssim_outside_mask_fallback_cv2(
    original: Image.Image,
    generated: Image.Image,
    inpaint_mask: "object",
) -> Optional[float]:
    """Compute SSIM on preserved region using an OpenCV implementation.

    This is a fallback for environments where scikit-image is missing/broken.
    Expects original and generated to have the same size.
    """
    try:
        import cv2
        import numpy as np
    except Exception:
        return None

    o = np.array(original.convert("L")).astype(np.float32) / 255.0
    g = np.array(generated.convert("L")).astype(np.float32) / 255.0

    if o.shape != g.shape:
        return None

    m = inpaint_mask
    if not isinstance(m, np.ndarray):
        try:
            m = np.array(m)
        except Exception:
            return None
    m = m.astype(np.float32)
    if m.max() > 1.0:
        m = m / 255.0

    outside = m < 0.5
    if not outside.any():
        return 1.0

    # Zero-out the inpaint region in both images, like calculate_ssim_masked.
    o2 = o.copy()
    g2 = g.copy()
    o2[~outside] = 0.0
    g2[~outside] = 0.0

    # SSIM (Wang et al.) using Gaussian window.
    # Constants for L=1.
    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)

    mu1 = cv2.GaussianBlur(o2, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(o2 * o2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(g2 * g2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(o2 * g2, (11, 11), 1.5) - mu1_mu2

    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

    ssim_map = numerator / (denominator + 1e-12)
    return float(ssim_map.mean())


def _find_default_inpainting_checkpoint(project_root: Path) -> Path:
    inpainting_dir = project_root / "models" / "inpainting"
    if not inpainting_dir.exists():
        raise FileNotFoundError(f"Inpainting directory not found: {inpainting_dir}")

    for name in ("final_model", "best_model"):
        candidate = inpainting_dir / name
        if candidate.exists() and candidate.is_dir():
            return candidate

    # Fall back to latest checkpoint_epoch_*
    epochs = []
    for p in inpainting_dir.iterdir():
        if p.is_dir() and p.name.startswith("checkpoint_epoch_"):
            try:
                epochs.append((int(p.name.split("_")[-1]), p))
            except Exception:
                continue
    if epochs:
        epochs.sort(key=lambda x: x[0], reverse=True)
        return epochs[0][1]

    raise FileNotFoundError(f"No checkpoints found under {inpainting_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create evaluation pictures with CTR + classifier overlay")

    p.add_argument("--config", type=str, default="configs/inference.yaml")
    p.add_argument("--input-dir", type=str, default="data/raw/cardiomegaly")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="LoRA checkpoint directory (default: models/inpainting/final_model, best_model, or latest checkpoint_epoch_*)",
    )
    p.add_argument("--output-dir", type=str, default="Evaluation_pictures")
    p.add_argument(
        "--num",
        type=int,
        default=None,
        help="Backward-compatible alias for --random-num (only used if best/worst are 0)",
    )
    p.add_argument("--best-num", type=int, default=4, help="How many best examples to save")
    p.add_argument("--worst-num", type=int, default=4, help="How many worst examples to save")
    p.add_argument("--random-num", type=int, default=4, help="How many random examples to save")
    p.add_argument(
        "--pool-success",
        type=int,
        default=None,
        help="How many successful candidates to collect before selecting best/worst/random (default: max(50, total*5))",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--classifier-calibration",
        type=str,
        default="outputs/classifier/dataset_a_calibration.json",
        help=(
            "Path to classifier calibration JSON (written by scripts/calibrate_classifier_temperature.py). "
            "Used only if --classifier-temperature is not set. Ignored if missing."
        ),
    )
    p.add_argument(
        "--disable-classifier-calibration",
        action="store_true",
        help="Disable loading temperature from calibration JSON (forces uncalibrated T=1.0 unless --classifier-temperature is provided)",
    )
    p.add_argument(
        "--classifier-temperature",
        type=float,
        default=None,
        help=(
            "Softmax temperature for classifier scores (overrides --classifier-calibration). "
            "Use T>1 to reduce overconfidence; ideally fit on val set."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed (omit for different random selection each run)",
    )
    p.add_argument("--save-comparison", action="store_true", help="Also save plain side-by-side comparison")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    rng = random.Random(args.seed)

    config = load_config(args.config)

    # Resolve classifier temperature: CLI override > calibration JSON > default.
    classifier_temperature: float = 1.0
    classifier_calibrated: bool = False
    temperature_source: str = "default"

    if args.classifier_temperature is not None:
        classifier_temperature = float(args.classifier_temperature)
        classifier_calibrated = classifier_temperature != 1.0
        temperature_source = "cli"
    elif not args.disable_classifier_calibration and args.classifier_calibration:
        calibration_path = resolve_path(args.classifier_calibration, PROJECT_ROOT)
        t = _try_load_temperature_from_calibration_json(calibration_path)
        if t is not None:
            classifier_temperature = float(t)
            classifier_calibrated = classifier_temperature != 1.0
            temperature_source = f"calibration_json:{calibration_path}"

    print(f"Classifier temperature: T={classifier_temperature:.4g} (source: {temperature_source})")

    input_dir = PROJECT_ROOT / args.input_dir
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mode selection:
    # --num is a deprecated alias for --random-num and is only applied when best/worst are disabled.
    # This avoids accidentally changing the default presentation behavior (4/4/4).
    if args.num is not None:
        if int(args.best_num) == 0 and int(args.worst_num) == 0:
            args.random_num = int(args.num)
        else:
            print("Warning: --num is ignored because --best-num/--worst-num are non-zero. Use --random-num instead.")

    # Imports here to avoid slow imports if args invalid
    from src.models.classifier import CardiomegalyClassifier
    from src.models.segmenter import ChexMaskSegmenter
    from src.models.inpainter import CardiacInpainter
    from src.validation.anatomical import AnatomicalValidator
    from src.inference.pipeline import CardiomegalyToHealthyPipeline
    try:
        from src.validation.metrics import calculate_ssim_masked
    except Exception:
        calculate_ssim_masked = None

    # Load models
    classifier_path = resolve_path(config.models.classifier_path, PROJECT_ROOT)
    classifier = CardiomegalyClassifier(
        weights_path=classifier_path,
        device=args.device,
        temperature=float(classifier_temperature),
    )

    weights_dir = resolve_path(config.models.chexmask_weights_dir, PROJECT_ROOT)
    segmenter = ChexMaskSegmenter(weights_dir=weights_dir, device=args.device)

    checkpoint_path = (
        resolve_path(args.checkpoint, PROJECT_ROOT)
        if args.checkpoint
        else _find_default_inpainting_checkpoint(PROJECT_ROOT)
    )
    inpainter = CardiacInpainter(
        base_model_id=config.models.base_inpainting_model,
        lora_weights_path=checkpoint_path,
        device=args.device,
    )

    validator = AnatomicalValidator(
        segmenter=segmenter,
        min_ctr=config.validation.min_ctr,
        max_ctr=config.validation.max_ctr,
    )

    pipeline = CardiomegalyToHealthyPipeline(
        classifier=classifier,
        segmenter=segmenter,
        inpainter=inpainter,
        validator=validator,
        config=config,
    )

    # Always compute classifier outputs; allow pipeline to use it too.
    pipeline.require_classifier = True
    pipeline.require_anatomical = True

    if calculate_ssim_masked is None:
        print("Warning: scikit-image SSIM unavailable; using OpenCV fallback SSIM.")
        print("  (Optional) To use skimage SSIM: fix/install scikit-image")

    # Mask settings for SSIM "outside" region (use config if available)
    dilation_factor = 0.1
    feather_radius = 5
    if hasattr(config, "mask"):
        try:
            dilation_factor = float(config.mask.get("dilation_factor", dilation_factor))
            feather_radius = int(config.mask.get("feather_radius", feather_radius))
        except Exception:
            pass

    # Collect images
    image_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not image_paths:
        raise RuntimeError(f"No images found in {input_dir}")

    # Randomize selection each run (unless seed provided).
    rng.shuffle(image_paths)

    total_needed = int(args.best_num) + int(args.worst_num) + int(args.random_num)
    if total_needed <= 0:
        raise ValueError("Nothing to do: best+worst+random must be > 0")

    pool_success_target = (
        int(args.pool_success)
        if args.pool_success is not None
        else max(50, total_needed * 5)
    )

    print(
        f"Pool target: {pool_success_target} successful candidates "
        f"(from {len(image_paths)} available input images)"
    )

    pool_dir = output_dir / "_pool"
    pool_dir.mkdir(parents=True, exist_ok=True)

    attempted = 0
    candidates: List[Dict] = []
    max_attempts_total = max(len(image_paths), pool_success_target * 10)

    # Create group dirs up-front so a partially completed run still has the expected structure.
    best_dir = output_dir / f"best{int(args.best_num)}"
    worst_dir = output_dir / f"worst{int(args.worst_num)}"
    random_dir = output_dir / f"random{int(args.random_num)}"
    for d, n in ((best_dir, args.best_num), (worst_dir, args.worst_num), (random_dir, args.random_num)):
        if int(n) > 0:
            d.mkdir(parents=True, exist_ok=True)

    def _write_summary_partial(results: List[Dict]) -> None:
        summary = {
            "timestamp": datetime.now().isoformat(),
            "input_dir": str(input_dir),
            "checkpoint": str(checkpoint_path),
            "seed": args.seed,
            "classifier_temperature": float(classifier_temperature),
            "classifier_temperature_source": temperature_source,
            "requested": {
                "best": int(args.best_num),
                "worst": int(args.worst_num),
                "random": int(args.random_num),
                "total": int(total_needed),
                "pool_success_arg": None if args.pool_success is None else int(args.pool_success),
                "pool_success_target": int(pool_success_target),
            },
            "saved": {
                "best": 0,
                "worst": 0,
                "random": 0,
                "total": int(len(results)),
                "pool_success": int(len(candidates)),
            },
            "attempted": int(attempted),
            "score_definition": "score = 0.40*P(healthy) + 0.40*SSIM_outside + 0.20*CTR<=max (penalize only above threshold)",
            "candidates": candidates,
            "selected": {"best": [], "worst": [], "random": []},
            "results": results,
            "note": "Partial summary (run may have been interrupted).",
        }
        try:
            (output_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, default=str), encoding="utf-8"
            )
        except Exception:
            pass

    try:
        for img_path in image_paths:
            if len(candidates) >= pool_success_target:
                break
            if attempted >= max_attempts_total:
                break

            attempted += 1

            original = _normalize_single_xray(Image.open(img_path).convert("L"))

            # Compute input metrics
            in_pred, in_conf = classifier.predict_with_confidence(original)
            in_ctr = _compute_ctr(segmenter, validator, original)

            # Run pipeline
            details = pipeline.process_with_details(img_path, skip_verification=True)
            if not details.get("success") or details.get("output") is None:
                continue

            generated = _normalize_single_xray(details["output"].convert("L"))
            # Resize generated to original size for metrics that assume identical geometry.
            generated_metrics = generated.resize(original.size, Image.Resampling.LANCZOS)

            # SSIM (outside inpaint mask) - how much non-heart region changed
            ssim_outside = None
            try:
                masks_for_ssim = segmenter.segment(original)
                inpaint_mask = segmenter.prepare_inpainting_mask(
                    masks_for_ssim["heart"],
                    dilation_factor=dilation_factor,
                    feather_radius=feather_radius,
                )

                if calculate_ssim_masked is not None:
                    try:
                        ssim_outside = float(
                            calculate_ssim_masked(original, generated_metrics, inpaint_mask).get("outside")
                        )
                    except Exception:
                        ssim_outside = None

                if ssim_outside is None:
                    ssim_outside = _ssim_outside_mask_fallback_cv2(original, generated_metrics, inpaint_mask)
            except Exception:
                ssim_outside = None

            # Compute output metrics (explicitly, to ensure they exist)
            out_pred, out_conf = classifier.predict_with_confidence(generated_metrics)
            out_ctr = _compute_ctr(segmenter, validator, generated_metrics)

            score = _composite_score(
                out_pred=out_pred,
                out_conf=out_conf,
                ssim_outside=ssim_outside,
                out_ctr=out_ctr,
                min_ctr=float(config.validation.min_ctr),
                max_ctr=float(config.validation.max_ctr),
            )

            title = img_path.name
            panel = _make_panel(
                original=original,
                generated=generated,
                input_ctr=in_ctr,
                output_ctr=out_ctr,
                input_pred=in_pred,
                input_conf=in_conf,
                output_pred=out_pred,
                output_conf=out_conf,
                ssim_outside=ssim_outside,
                title=title,
                classifier_temperature=float(classifier_temperature) if classifier_calibrated else None,
                classifier_calibrated=bool(classifier_calibrated),
            )

            out_path = pool_dir / f"{img_path.stem}_EVAL.png"
            panel.save(out_path)

            if args.save_comparison:
                # Plain comparison image for convenience
                pad = 10
                orig = _to_rgb_512(original)
                gen = _to_rgb_512(generated)
                comp = Image.new("RGB", (orig.width * 2 + pad, orig.height), (0, 0, 0))
                comp.paste(orig, (0, 0))
                comp.paste(gen, (orig.width + pad, 0))
                comp.save(output_dir / f"{img_path.stem}_COMPARISON.png")

            candidates.append(
                {
                    "image": str(img_path),
                    "pool_output": str(out_path),
                    "score": float(score),
                    "input": {"ctr": in_ctr, "pred": in_pred, "conf": in_conf},
                    "output_metrics": {
                        "ctr": out_ctr,
                        "pred": out_pred,
                        "conf": out_conf,
                        "ssim_outside": ssim_outside,
                    },
                }
            )

            # Persist progress so we don't end up with only _pool if the run stops.
            _write_summary_partial(results=[])

    except KeyboardInterrupt:
        print("Interrupted by user; selecting best/worst/random from collected candidates...")

    if len(candidates) < pool_success_target:
        print(
            f"Warning: Only collected {len(candidates)}/{pool_success_target} successful candidates. "
            "This usually means many attempts failed (e.g., pipeline returned success=False)."
        )

    # Select best/worst/random from candidates.
    # Clean only the group folders (not pool) to keep the output fresh.
    for d, n in ((best_dir, args.best_num), (worst_dir, args.worst_num), (random_dir, args.random_num)):
        if int(n) <= 0:
            continue
        for p in d.glob("*_EVAL.png"):
            try:
                p.unlink()
            except Exception:
                pass

    candidates_sorted_desc = sorted(candidates, key=lambda r: r.get("score", 0.0), reverse=True)
    candidates_sorted_asc = list(reversed(candidates_sorted_desc))

    selected_best = candidates_sorted_desc[: int(args.best_num)] if args.best_num > 0 else []
    selected_worst = candidates_sorted_asc[: int(args.worst_num)] if args.worst_num > 0 else []

    used = {r["pool_output"] for r in (selected_best + selected_worst)}
    remaining = [r for r in candidates if r["pool_output"] not in used]
    rng.shuffle(remaining)
    selected_random = remaining[: int(args.random_num)]

    # If we didn't have enough remaining, top up from the full candidate set.
    if len(selected_random) < int(args.random_num):
        needed = int(args.random_num) - len(selected_random)
        fallback = [r for r in candidates if r["pool_output"] not in {x["pool_output"] for x in selected_random}]
        rng.shuffle(fallback)
        selected_random.extend(fallback[:needed])

    def _copy_selected(selected: List[Dict], out_dir: Path, tag: str) -> List[Dict]:
        out: List[Dict] = []
        for i, r in enumerate(selected, start=1):
            src = Path(r["pool_output"])
            dst = out_dir / f"{tag}{i:02d}_{Path(r['image']).stem}_EVAL.png"
            try:
                shutil.copy2(src, dst)
            except Exception:
                # If copy fails, still record but point to pool output.
                dst = src
            rr = dict(r)
            rr["output"] = str(dst)
            rr["group"] = tag.rstrip("_")
            out.append(rr)
        return out

    best_out = _copy_selected(selected_best, best_dir, "best_") if int(args.best_num) > 0 else []
    worst_out = _copy_selected(selected_worst, worst_dir, "worst_") if int(args.worst_num) > 0 else []
    random_out = _copy_selected(selected_random, random_dir, "random_") if int(args.random_num) > 0 else []

    # Convenience: keep a flat list in display order
    results = best_out + worst_out + random_out

    summary = {
        "timestamp": datetime.now().isoformat(),
        "input_dir": str(input_dir),
        "checkpoint": str(checkpoint_path),
        "seed": args.seed,
        "classifier_temperature": float(classifier_temperature),
        "classifier_temperature_source": temperature_source,
        "requested": {
            "best": int(args.best_num),
            "worst": int(args.worst_num),
            "random": int(args.random_num),
            "total": int(total_needed),
            "pool_success_arg": None if args.pool_success is None else int(args.pool_success),
            "pool_success_target": int(pool_success_target),
        },
        "saved": {
            "best": int(len(best_out)),
            "worst": int(len(worst_out)),
            "random": int(len(random_out)),
            "total": int(len(results)),
            "pool_success": int(len(candidates)),
        },
        "attempted": int(attempted),
        "score_definition": "score = 0.40*P(healthy) + 0.40*SSIM_outside + 0.20*CTR<=max (penalize only above threshold)",
        "candidates": candidates,
        "selected": {"best": best_out, "worst": worst_out, "random": random_out},
        "results": results,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print(
        f"Saved best={len(best_out)} worst={len(worst_out)} random={len(random_out)} (pool_success={len(candidates)}) to: {output_dir}"
    )


if __name__ == "__main__":
    main()
