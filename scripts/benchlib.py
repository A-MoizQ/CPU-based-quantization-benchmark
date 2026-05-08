import json
import os
import platform
import resource
import threading
import time
from pathlib import Path


try:
    import psutil
except ImportError:  # pragma: no cover - fallback for minimal installs
    psutil = None


def bytes_to_mib(value):
    return value / (1024 ** 2)


def bytes_to_gib(value):
    return value / (1024 ** 3)


def directory_size_bytes(path):
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(path, record):
    ensure_parent(path)
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_json(path, payload):
    ensure_parent(path)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def read_jsonl(path, limit=None):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def current_rss_bytes(pid=None):
    if psutil is not None:
        proc = psutil.Process(pid or os.getpid())
        return proc.memory_info().rss
    # ru_maxrss is KiB on Linux, bytes on macOS. This project targets Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


class MemorySampler:
    def __init__(self, pid=None, interval=0.05):
        self.pid = pid or os.getpid()
        self.interval = interval
        self.start_rss = 0
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        self.start_rss = current_rss_bytes(self.pid)
        self.peak_rss = self.start_rss
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self.peak_rss = max(self.peak_rss, current_rss_bytes(self.pid))
        except Exception:
            pass

    def _sample(self):
        while not self._stop.is_set():
            try:
                self.peak_rss = max(self.peak_rss, current_rss_bytes(self.pid))
            except Exception:
                pass
            time.sleep(self.interval)

    @property
    def peak_delta_bytes(self):
        return max(0, self.peak_rss - self.start_rss)


def system_info():
    cpu_name = platform.processor() or platform.machine()
    if psutil is not None:
        vm = psutil.virtual_memory()
        total_ram = vm.total
        available_ram = vm.available
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
    else:
        total_ram = None
        available_ram = None
        physical_cores = None
        logical_cores = os.cpu_count()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu_name,
        "physical_cores": physical_cores,
        "logical_cores": logical_cores,
        "ram_total_mib": bytes_to_mib(total_ram) if total_ram else None,
        "ram_available_mib": bytes_to_mib(available_ram) if available_ram else None,
    }


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
