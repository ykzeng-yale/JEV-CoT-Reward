#!/usr/bin/env python3
"""Read-only hardware, local runtime, and model-cache metadata inventory.

Never reads credentials, environment values, model tensor contents, or hardware
serial numbers. No packages are installed and no network calls are made unless
the caller explicitly supplies a known SSH host. Standard library only.
"""

import argparse
import datetime
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def command(args, timeout=15, input_text=None):
    try:
        p = subprocess.run(args, input=input_text, capture_output=True, text=True, timeout=timeout)
        return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": type(exc).__name__, "detail": str(exc)}


def sysctl(key):
    result = command(["sysctl", "-n", key])
    if result.get("returncode") == 0:
        text = result["stdout"]
        return int(text) if text.isdigit() else text
    return None


def model_inventory(cache):
    models = []
    for model in sorted(cache.glob("models--*")):
        snapshots = []
        for snap in sorted((model / "snapshots").glob("*")):
            weights = []
            for path in sorted(snap.glob("*")):
                if path.is_file() and path.suffix in {".safetensors", ".gguf", ".bin"}:
                    weights.append({"name": path.name, "bytes": path.stat().st_size})
            snapshots.append({"path": str(snap), "weights": weights,
                              "weight_bytes": sum(w["bytes"] for w in weights)})
        models.append({"cache_name": model.name, "snapshots": snapshots})
    return models


def runtime_inventory(python):
    source = '''import importlib.metadata as m, json, sys
p = {}
for name in ["mlx", "mlx-lm", "torch", "transformers", "numpy", "scikit-learn", "llama-cpp-python"]:
    try: p[name] = m.version(name)
    except m.PackageNotFoundError: p[name] = None
print(json.dumps({"python": sys.version.split()[0], "executable": sys.executable, "packages": p}))'''
    result = command([str(python), "-c", source])
    if result.get("returncode") == 0:
        return json.loads(result["stdout"])
    return {"executable": str(python), "error": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--python", action="append", default=[])
    parser.add_argument("--ssh-host", help="Explicit known host only; strict host key checking, no prompts")
    args = parser.parse_args()
    disk = shutil.disk_usage(Path.cwd())
    result = {
        "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "chip": sysctl("machdep.cpu.brand_string"),
        "ram_bytes": sysctl("hw.memsize"),
        "logical_cpus": sysctl("hw.logicalcpu"),
        "disk": {"path": str(Path.cwd()), "total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
        "swap": sysctl("vm.swapusage"),
        "runtime_inventory": [runtime_inventory(p) for p in dict.fromkeys([sys.executable] + args.python)],
        "huggingface_cache": model_inventory(Path.home() / ".cache/huggingface/hub"),
        "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
    }
    # Read only display metadata and retain only public hardware characteristics.
    if platform.system() == "Darwin":
        display = command(["system_profiler", "SPDisplaysDataType", "-json"])
        if display.get("returncode") == 0:
            data = json.loads(display["stdout"])
            result["gpus"] = [{k: gpu[k] for k in ("sppci_model", "sppci_cores", "spdisplays_metal") if k in gpu}
                              for gpu in data.get("SPDisplaysDataType", [])]
    if args.ssh_host:
        if args.ssh_host.startswith("-"):
            parser.error("SSH host must not begin with an option")
        # Does not add trust entries, supply passwords, or change the remote host.
        remote_command = "uname -s; uname -m; sysctl -n machdep.cpu.brand_string hw.memsize hw.logicalcpu; df -k /"
        result["remote"] = {"host": args.ssh_host, "probe": command([
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o",
            "StrictHostKeyChecking=yes", args.ssh_host, remote_command], timeout=12)}
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(str(args.output))
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
