"""Hardware stamp.

Every metrics JSON in this study must carry the hardware it was measured on.
Timings from different stamps must never be compared in the same table.
"""

import json
import platform
import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"


def _cpu_model() -> str:
    try:
        text = Path("/proc/cpuinfo").read_text()
        match = re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
        if match:
            return match.group(1).strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _nvidia_smi() -> dict:
    query = "name,memory.total,driver_version"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {"gpu_model": None, "gpu_memory_mib": None, "driver_version": None}
    if not out:
        return {"gpu_model": None, "gpu_memory_mib": None, "driver_version": None}
    name, mem, driver = [p.strip() for p in out.splitlines()[0].split(",")]
    return {
        "gpu_model": name,
        "gpu_memory_mib": int(mem.replace("MiB", "").strip()),
        "driver_version": driver,
    }


def _torch_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"torch_version": None, "cuda_version": None, "cuda_available": False}
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }


def hardware_stamp(device: str = "cpu") -> dict:
    """Return the hardware block. `device` is what the model actually ran on."""
    import psutil

    stamp = {
        "cpu_model": _cpu_model(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 2),
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "device_used": device,
    }
    stamp.update(_nvidia_smi())
    stamp.update(_torch_info())
    return stamp


def stamp_fingerprint(stamp: dict) -> str:
    """Stable identity of the machine. If this changes, all timings must be re-run."""
    keys = ["cpu_model", "cpu_cores_logical", "ram_total_gb", "gpu_model",
            "gpu_memory_mib", "driver_version"]
    return "|".join(str(stamp.get(k)) for k in keys)


def check_or_write_reference(stamp: dict) -> None:
    """Compare against the frozen reference stamp. Raise if the machine changed."""
    ref_path = METRICS_DIR / "hardware_reference.json"
    fingerprint = stamp_fingerprint(stamp)
    if not ref_path.exists():
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref_path.write_text(json.dumps(
            {"fingerprint": fingerprint, "stamp": stamp}, indent=2))
        return
    ref = json.loads(ref_path.read_text())
    if ref["fingerprint"] != fingerprint:
        raise RuntimeError(
            "HARDWARE CHANGED. All timing numbers are now invalid.\n"
            f"  reference: {ref['fingerprint']}\n"
            f"  current:   {fingerprint}\n"
            "Re-run every timing phase on this machine before building the "
            "comparison table."
        )


def save_metrics(name: str, payload: dict, device: str = "cpu") -> Path:
    """Write reports/metrics/<name>.json with the hardware block attached."""
    stamp = hardware_stamp(device=device)
    check_or_write_reference(stamp)
    payload = dict(payload)
    payload["hardware"] = stamp
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


if __name__ == "__main__":
    s = hardware_stamp(device="cpu")
    check_or_write_reference(s)
    print(json.dumps(s, indent=2))
