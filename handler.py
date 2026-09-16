"""
RunPod Serverless Handler — FaceFusion HyperSwap 256 Face Swap

Uses official facefusion/facefusion:3.6.0-cuda image.
Same input/output format as inswapper: source_image + target_image → swapped image as base64.
"""

import os
import sys
import uuid
import base64
import time
import subprocess
import runpod

WORKSPACE = "/tmp/facefusion_jobs"
FACEFUSION_DIR = "/opt/facefusion"

os.makedirs(WORKSPACE, exist_ok=True)


def _detect_ext(data: bytes) -> str:
    """Guess a sensible file extension from magic bytes (FaceFusion checks extensions)."""
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return ".jpg"


def decode_base64_image(b64_string, dir_path, name):
    """Decode base64 (with or without data URI prefix) to a file, preserving format."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    data = base64.b64decode(b64_string)
    path = os.path.join(dir_path, name + _detect_ext(data))
    with open(path, "wb") as f:
        f.write(data)
    return path


def encode_image_base64(path):
    """Read image file and return raw base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _clamp(value, lo, hi, default):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def handler(job):
    t0 = time.time()
    job_input = job.get("input") or {}

    source_image = job_input.get("source_image")
    target_image = job_input.get("target_image")

    if not source_image or not target_image:
        return {"error": "source_image and target_image are required"}

    # Config — compatible with existing worker params
    model = job_input.get("model", "hyperswap_1c_256")
    face_restore = job_input.get("face_restore", True)
    face_enhancer_model = job_input.get("face_enhancer_model", "codeformer")
    # CodeFormer fidelity maps to FaceFusion's --face-enhancer-weight (0..1).
    codeformer_fidelity = _clamp(job_input.get("codeformer_fidelity", 0.7), 0.0, 1.0, 0.7)
    # How strongly the enhanced face is blended over the original (0..100).
    face_enhancer_blend = int(_clamp(job_input.get("face_enhancer_blend", 80), 0, 100, 80))
    detector_score = _clamp(job_input.get("face_detector_score", 0.3), 0.0, 1.0, 0.3)
    output_quality = int(_clamp(job_input.get("output_image_quality", 95), 1, 100, 95))
    pixel_boost = job_input.get("pixel_boost")  # e.g. "256x256", "512x512"; optional
    timeout_s = int(_clamp(job_input.get("timeout", 180), 10, 600, 180))

    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(WORKSPACE, job_id)
    os.makedirs(job_dir, exist_ok=True)

    source_path = target_path = None
    output_path = os.path.join(job_dir, "output.jpg")

    try:
        source_path = decode_base64_image(source_image, job_dir, "source")
        target_path = decode_base64_image(target_image, job_dir, "target")

        # Build FaceFusion 3.6 headless-run command.
        cmd = [
            sys.executable, os.path.join(FACEFUSION_DIR, "facefusion.py"),
            "headless-run",
            "-s", source_path,
            "-t", target_path,
            "-o", output_path,
            "--processors", "face_swapper",
            "--face-swapper-model", model,
            "--face-detector-model", "yolo_face",
            "--face-detector-score", str(detector_score),
            "--output-image-quality", str(output_quality),
            "--execution-providers", "cuda",
            # Models ship in the image; 'lite' verifies only what is missing.
            # (--skip-download was removed in FaceFusion 3.6.)
            "--download-scope", "lite",
        ]

        if pixel_boost:
            cmd.extend(["--face-swapper-pixel-boost", str(pixel_boost)])

        if face_restore:
            # --processors takes multiple values; add the enhancer as its own token
            # right after "face_swapper" so both run in one pass.
            proc_idx = cmd.index("--processors")
            cmd.insert(proc_idx + 2, "face_enhancer")
            cmd.extend([
                "--face-enhancer-model", face_enhancer_model,
                "--face-enhancer-weight", str(codeformer_fidelity),
                "--face-enhancer-blend", str(face_enhancer_blend),
            ])

        print(f"[HyperSwap] Job {job_id}: FaceFusion {model} "
              f"(enhancer={'on' if face_restore else 'off'})")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=FACEFUSION_DIR,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
        )

        if result.returncode != 0:
            stderr = (result.stderr or "")[-1000:]
            stdout = (result.stdout or "")[-1000:]
            print(f"[HyperSwap] FAILED (exit {result.returncode})")
            print(f"[HyperSwap] stderr: {stderr}")
            print(f"[HyperSwap] stdout: {stdout}")
            return {"error": f"FaceFusion exit {result.returncode}: {stderr[-300:] or stdout[-300:]}"}

        if not os.path.exists(output_path):
            return {"error": f"No output file. stdout: {(result.stdout or '')[-300:]}"}

        image_b64 = encode_image_base64(output_path)
        elapsed = time.time() - t0
        size_mb = len(image_b64) * 0.75 / 1024 / 1024

        print(f"[HyperSwap] Done in {elapsed:.1f}s ({size_mb:.1f}MB)")

        return {"image": image_b64}

    except subprocess.TimeoutExpired:
        print(f"[HyperSwap] Job {job_id} timed out after {timeout_s}s")
        return {"error": f"FaceFusion timed out after {timeout_s}s"}

    finally:
        for f in (source_path, target_path, output_path):
            if not f:
                continue
            try:
                os.remove(f)
            except OSError:
                pass
        try:
            os.rmdir(job_dir)
        except OSError:
            pass


runpod.serverless.start({"handler": handler})
