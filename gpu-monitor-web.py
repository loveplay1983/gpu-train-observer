# usage : python gpu-monitor-web.py --host 127.0.0.1 --port 5000 --interval 2 --log gpu_monitor_log.csv 
#########################################################################################################

from flask import Flask, render_template_string, send_file, request, jsonify, redirect, url_for, render_template
import threading
import time
import csv
import os
from datetime import datetime, timedelta
from collections import deque
from typing import List, Dict, Any, Optional

# Telemetry libs
try:
    import pynvml
except Exception:
    pynvml = None

try:
    import psutil
except Exception:
    psutil = None

# Optional plotting and dataframe
try:
    import pandas as pd
    import matplotlib.pyplot as plt
except Exception:
    pd = None
    plt = None

# Optional torch
try:
    import torch
except Exception:
    torch = None

# ----------------------------
# Configuration defaults
# ----------------------------
DEFAULT_LOG = "gpu_monitor_log.csv"
DEFAULT_INTERVAL = 2  # seconds
WARN_THRESHOLD = 0.85  # fraction of total VRAM to warn (85%)
TREND_WINDOW = 10  # number of recent samples to compute growth trend
TREND_MB_PER_SEC_THRESHOLD = 50.0  # MB/sec growth considered alarming

# ----------------------------
# Globals (shared state)
# ----------------------------
monitor_thread: Optional[threading.Thread] = None
monitor_stop_event = threading.Event()
monitor_lock = threading.Lock()
is_monitoring = False

# Recent samples buffer for anomaly detection (per GPU)
recent_samples: Dict[int, deque] = {}  # gpu_index -> deque of (timestamp, nvml_memory_used_MB)

# In-memory latest status for UI
latest_status: Dict[str, Any] = {"running": False, "last_sample": None, "alerts": []}

# Ensure CSV header columns are stable
CSV_FIELDS = [
    "timestamp", "gpu_index", "gpu_name",
    "nvml_memory_used_MB", "nvml_memory_total_MB",
    "nvml_gpu_util_%", "nvml_mem_util_%", "nvml_temperature_C",
    "pytorch_allocated_MB", "pytorch_reserved_MB", "pytorch_fragmentation",
    "cpu_percent", "ram_used_GB", "ram_total_GB", "alerts"
]


# ----------------------------
# NVML and system collection
# ----------------------------
def init_nvml_safe() -> bool:
    if pynvml is None:
        return False
    try:
        pynvml.nvmlInit()
        return True
    except Exception:
        return False


def shutdown_nvml_safe():
    if pynvml:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def collect_nvml() -> List[Dict[str, Any]]:
    if pynvml is None:
        return []
    rows = []
    count = pynvml.nvmlDeviceGetCount()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    for i in range(count):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        # Safe decode: only decode if bytes
        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode("utf-8")

        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        try:
            temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            temp = None
        rows.append({
            "timestamp": ts,
            "gpu_index": i,
            "gpu_name": name,
            "nvml_memory_used_MB": round(mem.used / 1024**2, 2),
            "nvml_memory_total_MB": round(mem.total / 1024**2, 2),
            "nvml_gpu_util_%": util.gpu,
            "nvml_mem_util_%": util.memory,
            "nvml_temperature_C": temp,
        })
    return rows



def collect_pytorch_snapshot() -> List[Dict[str, Any]]:
    """Collect PyTorch allocator stats for the current process."""
    if torch is None or not torch.cuda.is_available():
        return []
    rows = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    dev_count = torch.cuda.device_count()
    for i in range(dev_count):
        try:
            torch.cuda.synchronize(i)
        except Exception:
            pass
        alloc = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        frag = None
        if reserved > 0:
            frag = round(1.0 - (alloc / reserved), 4)
        rows.append({
            "timestamp": ts,
            "gpu_index": i,
            "pytorch_allocated_MB": round(alloc / 1024**2, 2),
            "pytorch_reserved_MB": round(reserved / 1024**2, 2),
            "pytorch_fragmentation": frag
        })
    return rows


def collect_system() -> Dict[str, Any]:
    if psutil is None:
        return {"cpu_percent": None, "ram_used_GB": None, "ram_total_GB": None}
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()
    return {"cpu_percent": cpu, "ram_used_GB": round(ram.used / 1024**3, 2), "ram_total_GB": round(ram.total / 1024**3, 2)}


# ----------------------------
# Merge and log helpers
# ----------------------------
def merge_rows(nvml_rows: List[Dict[str, Any]], pyt_rows: List[Dict[str, Any]], sys_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    pyt_by_idx = {r["gpu_index"]: r for r in pyt_rows}
    merged = []
    for r in nvml_rows:
        idx = r["gpu_index"]
        m = {**r}
        pyt = pyt_by_idx.get(idx, {})
        m["pytorch_allocated_MB"] = pyt.get("pytorch_allocated_MB")
        m["pytorch_reserved_MB"] = pyt.get("pytorch_reserved_MB")
        m["pytorch_fragmentation"] = pyt.get("pytorch_fragmentation")
        m.update(sys_row)
        m["alerts"] = ""  # will fill with anomaly detector messages
        merged.append(m)
    return merged


def append_csv(rows: List[Dict[str, Any]], filepath: str):
    if not rows:
        return
    file_exists = os.path.isfile(filepath)
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        for r in rows:
            out = {k: ("" if r.get(k) is None else r.get(k)) for k in CSV_FIELDS}
            writer.writerow(out)


# ----------------------------
# Anomaly detection
# ----------------------------
def update_recent_samples_and_detect(rows: List[Dict[str, Any]]) -> List[str]:
    """Update sliding window and run anomaly checks.
    Returns list of alerts (strings) attached to this set of rows.
    """
    alerts = []
    for r in rows:
        idx = r["gpu_index"]
        used = r["nvml_memory_used_MB"]
        total = r["nvml_memory_total_MB"]
        # init deque
        if idx not in recent_samples:
            recent_samples[idx] = deque(maxlen=TREND_WINDOW)
        # push sample
        now_ts = datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S.%f")
        recent_samples[idx].append((now_ts, used))

        # check absolute threshold
        if total and used / total >= WARN_THRESHOLD:
            msg = f"GPU {idx} high memory usage: {used:.0f}/{total:.0f} MB ({used/total:.0%})"
            alerts.append(msg)

        # compute trend (MB/s) using earliest and latest in window if enough samples
        dq = recent_samples[idx]
        if len(dq) >= 3:
            t0, v0 = dq[0]
            t1, v1 = dq[-1]
            dt = (t1 - t0).total_seconds()
            if dt > 0:
                slope = (v1 - v0) / dt  # MB per second
                if slope >= TREND_MB_PER_SEC_THRESHOLD:
                    msg = f"GPU {idx} memory growing fast: {slope:.1f} MB/s over last {dt:.1f}s"
                    alerts.append(msg)
    # de-duplicate alerts
    unique_alerts = []
    [unique_alerts.append(a) for a in alerts if a not in unique_alerts]
    return unique_alerts


# ----------------------------
# Monitor thread loop
# ----------------------------
def monitor_loop(interval: int, logfile: str, include_pytorch: bool):
    global is_monitoring, latest_status
    if not init_nvml_safe():
        latest_status["running"] = False
        latest_status["alerts"] = ["NVML not available (pynvml missing or init failed)."]
        return

    latest_status["running"] = True
    monitor_stop_event.clear()
    try:
        while not monitor_stop_event.is_set():
            nvml_rows = collect_nvml()
            pyt_rows = collect_pytorch_snapshot() if include_pytorch else []
            sys_row = collect_system()
            merged = merge_rows(nvml_rows, pyt_rows, sys_row)
            # detect anomalies
            alerts = update_recent_samples_and_detect(merged)
            # attach alerts to rows and latest_status
            for r in merged:
                r["alerts"] = " | ".join(alerts) if alerts else ""
            latest_status["last_sample"] = merged
            latest_status["alerts"] = alerts
            append_csv(merged, logfile)
            # small print to server logs (optional)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] logged {len(merged)} GPU rows; alerts: {alerts}")
            time.sleep(interval)
    except Exception as e:
        latest_status["alerts"].append(f"Monitor error: {e}")
    finally:
        shutdown_nvml_safe()
        latest_status["running"] = False


# ----------------------------
# Flask app & endpoints
# ----------------------------
app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    log = request.args.get("log", DEFAULT_LOG)
    interval = request.args.get("interval", DEFAULT_INTERVAL)
    include_pytorch = request.args.get("include_pytorch", "false").lower() in ("1", "true", "yes")
    last_sample = latest_status.get("last_sample")
    last_time = None
    if last_sample:
        try:
            last_time = last_sample[0]["timestamp"]
        except Exception:
            last_time = None
    return render_template(
        "index.html",
        log=log, interval=interval, include_pytorch=include_pytorch,
        status=latest_status, last_sample=last_sample, last_time=last_time,
        alerts=latest_status.get("alerts", [])
    )

@app.route("/control", methods=["POST"])
def control():
    global monitor_thread, is_monitoring
    action = request.form.get("action")
    log = request.form.get("log", DEFAULT_LOG)
    interval = int(request.form.get("interval", DEFAULT_INTERVAL))
    include_pytorch = request.form.get("include_pytorch") is not None

    if action == "start":
        if monitor_thread and monitor_thread.is_alive():
            return redirect(url_for("index", log=log))
        monitor_thread = threading.Thread(target=monitor_loop, args=(interval, log, include_pytorch), daemon=True)
        monitor_thread.start()
        return redirect(url_for("index", log=log))
    elif action == "stop":
        monitor_stop_event.set()
        return redirect(url_for("index", log=log))
    elif action == "download":
        return redirect(url_for("download", log=log))
    else:
        return redirect(url_for("index", log=log))


@app.route("/download", methods=["GET"])
def download():
    log = request.args.get("log", DEFAULT_LOG)
    if not os.path.isfile(log):
        return f"Log file not found: {log}", 404
    return send_file(log, as_attachment=True)


@app.route("/plot", methods=["GET"])
def plot():
    log = request.args.get("log", DEFAULT_LOG)
    gpu_index = int(request.args.get("gpu", 0))
    savepath = f"/tmp/gpu_plot_{os.getpid()}_{gpu_index}.png"
    try:
        make_plot(log, gpu_index, savepath)
    except Exception as e:
        return f"Plot error: {e}", 500
    return send_file(savepath, mimetype="image/png")


@app.route("/status_json", methods=["GET"])
def status_json():
    return jsonify(latest_status)


# ----------------------------
# Plot generation helper (matplotlib)
# ----------------------------
def make_plot(logfile: str, gpu_index: int, outpath: str):
    if pd is None or plt is None:
        raise RuntimeError("pandas and matplotlib required for plotting.")
    if not os.path.isfile(logfile):
        raise FileNotFoundError("Log file not found: " + logfile)
    df = pd.read_csv(logfile)
    df = df[df["gpu_index"] == gpu_index].copy()
    if df.empty:
        raise ValueError("No data for gpu_index=" + str(gpu_index))
    # parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    # build plot
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df["timestamp"], df["nvml_memory_used_MB"], label="NVML Mem Used (MB)", linewidth=1.6)
    if "pytorch_allocated_MB" in df.columns:
        ax1.plot(df["timestamp"], df["pytorch_allocated_MB"], label="PyTorch Alloc (MB)", linestyle="--")
    ax1.set_ylabel("Memory MB")
    ax2 = ax1.twinx()
    ax2.plot(df["timestamp"], df["nvml_gpu_util_%"], label="GPU Util %", color="tab:red", alpha=0.7)
    ax2.plot(df["timestamp"], df["nvml_temperature_C"], label="Temp (°C)", color="tab:orange", alpha=0.7)
    # legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close(fig)


# ----------------------------
# Utility: append a single snapshot from inside training loop (no heavy overhead)
# ----------------------------
def log_pytorch_snapshot_to_csv(logfile: str = DEFAULT_LOG):
    """Convenience function for training script to append one snapshot with PyTorch stats.
    Call from training process to capture accurate allocator stats without running server's monitor.
    """
    if pynvml is None:
        raise RuntimeError("pynvml not installed.")
    if not init_nvml_safe():
        raise RuntimeError("Could not init NVML.")
    try:
        nvml_rows = collect_nvml()
        pyt_rows = collect_pytorch_snapshot()
        sys_row = collect_system()
        merged = merge_rows(nvml_rows, pyt_rows, sys_row)
        alerts = update_recent_samples_and_detect(merged)
        for r in merged:
            r["alerts"] = " | ".join(alerts) if alerts else ""
        append_csv(merged, logfile)
    finally:
        shutdown_nvml_safe()


# ----------------------------
# Main CLI run
# ----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GPU Monitor Flask UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)
    parser.add_argument("--log", default=DEFAULT_LOG)
    parser.add_argument("--include-pytorch", action="store_true",
                        help="If set and torch is importable in this process, include PyTorch allocator stats")
    args = parser.parse_args()

    # ensure header of CSV if not exists
    if not os.path.isfile(args.log):
        with open(args.log, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()

    # Start monitor thread immediately (optional) - comment if you prefer manual start
    monitor_thread = threading.Thread(target=monitor_loop, args=(args.interval, args.log, args.include_pytorch), daemon=True)
    monitor_thread.start()

    print(f"Starting Flask server on {args.host}:{args.port} ...")
    app.run(host=args.host, port=args.port, threaded=True)