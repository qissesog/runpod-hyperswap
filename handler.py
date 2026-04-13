"""
RunPod Serverless Handler — FaceFusion HyperSwap 256 Face Swap

Drop-in replacement for the inswapper RunPod worker.
Same input/output format: source_image + target_image → swapped image as base64.
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


def decode_base64_image(b64_string, path):
    """Decode base64 (with or without data URI prefix) to file."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64_string))


def encode_image_base64(path):
    """Read image file and return raw base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def handler(job):
    t0 = time.time()
    job_input = job["input"]

    source_image = job_input.get("source_image")
    target_image = job_input.get("target_image")

    if not source_image or not target_image:
        return {"error": "source_image and target_image are required"}

    # Config — compatible with existing worker params
    model = job_input.get("model", "hyperswap_1c_256")
    face_restore = job_input.get("face_restore", True)
    face_enhancer_model = job_input.get("face_enhancer_model", "codeformer")  # "codeformer" or "gfpgan_1.4"
    codeformer_fidelity = job_input.get("codeformer_fidelity", 0.7)

    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(WORKSPACE, job_id)
    os.makedirs(job_dir, exist_ok=True)

    source_path = os.path.join(job_dir, "source.jpg")
    target_path = os.path.join(job_dir, "target.jpg")
    output_path = os.path.join(job_dir, "output.jpg")

    try:
        decode_base64_image(source_image, source_path)
        decode_base64_image(target_image, target_path)

        # Build FaceFusion headless-run command
        cmd = [
            sys.executable, os.path.join(FACEFUSION_DIR, "facefusion.py"),
            "headless-run",
            "-s", source_path,
            "-t", target_path,
            "-o", output_path,
            "--processors", "face_swapper",
            "--face-swapper-model", model,
            "--face-detector-model", "yolo_face",
            "--face-detector-score", "0.3",
            "--output-image-quality", "95",
            "--execution-providers", "cuda",
        ]

        # Add face enhancer if requested
        if face_restore:
            # Insert face_enhancer after face_swapper in processors list
            proc_idx = cmd.index("--processors") + 1
            cmd.insert(proc_idx + 1, "face_enhancer")
            cmd.extend([
                "--face-enhancer-model", face_enhancer_model,
                "--face-enhancer-blend", str(int(codeformer_fidelity * 100)),
            ])

        print(f"[HyperSwap] Job {job_id}: running FaceFusion...")
        print(f"[HyperSwap] CMD: {' '.join(cmd[:15])}...")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=FACEFUSION_DIR,
        )

        if result.returncode != 0:
            stderr = result.stderr[-500:] if result.stderr else ""
            stdout = result.stdout[-500:] if result.stdout else ""
            print(f"[HyperSwap] FAILED (exit {result.returncode})")
            print(f"[HyperSwap] stderr: {stderr}")
            print(f"[HyperSwap] stdout: {stdout}")
            return {"error": f"FaceFusion exit {result.returncode}: {stderr[-200:]}"}

        if not os.path.exists(output_path):
            return {"error": f"No output file. stdout: {result.stdout[-200:]}"}

        image_b64 = encode_image_base64(output_path)
        elapsed = time.time() - t0
        size_mb = len(image_b64) * 0.75 / 1024 / 1024

        print(f"[HyperSwap] Done in {elapsed:.1f}s ({size_mb:.1f}MB)")

        return {"image": image_b64}

    finally:
        for f in [source_path, target_path, output_path]:
            try:
                os.remove(f)
            except OSError:
                pass
        try:
            os.rmdir(job_dir)
        except OSError:
            pass


runpod.serverless.start({"handler": handler})
