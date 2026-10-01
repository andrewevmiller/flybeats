"""Live loss chart for a run in progress, in the browser. Reads the run's per-step log as it grows.

memorise.py starts it by itself (LiveChart, below) once its new log is open, so there is nothing else to run: it
prints the local address, the first free port from 8765 (a second run side by side gets 8766). It can also run on its
own, for a run started without it:
    .venv\\Scripts\\python.exe scripts\\live_chart.py                        # memorisation run "real"
    .venv\\Scripts\\python.exe scripts\\live_chart.py --memorise control-1   # reports/memorise/control-1/
    .venv\\Scripts\\python.exe scripts\\live_chart.py --run baby_real        # train.py: runs/baby_real/log.jsonl
The page refreshes every 5 s. It also shows CPU and GPU readings (scripts/machine.py), taken every 10 s at idle
priority and saved next to the run's log, with a graph of them over the run. Reads the logs, writes the machine log.
Inside a run it costs the training loop next to nothing: a few milliseconds of Python a minute, and on each page
refresh while the local page is open.

The website shows one page: the run going now (or the last one to run). runs/_live_site/ (index.html, data.json) is
written every 60 s and copied to the site by the upload command ({dir} is the folder, {name} the run's name if wanted).
Each new run takes the page over; its title says which run it is ("Memorisation check · control 1"). The page reloads
its data every 30 s and says Offline when updates stop. When the run ends, however it ends, memorise.py sends one last
update marked final, so the website keeps showing how the run ended. The command lives in config/live.local.yaml
(this machine only, not committed; see load_local); without it the chart stays local. On its own, add --publish to
update the website.

Here: rclone, remote flybeats = an FTP account on Hostinger whose directory is public_html/flybeats-live (explicit TLS;
no_check_certificate because the server's certificate names *.hstgr.io, not the IP rclone connects to), with
    upload: rclone copy {dir} flybeats: --max-depth 1 --ignore-times --exclude *.tmp
--ignore-times re-sends the files every time (about 60 KB), so a server that keeps no file times cannot leave a stale
data.json on the site; rclone uploads under a temporary name and renames, so a half-sent file is never served.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flybeats.config import ROOT, reports_dir  # noqa: E402
from machine import ROW, Sampler  # noqa: E402

PAGE = Path(__file__).resolve().with_name("live_chart.html")
SITE_ROOT = ROOT / "runs" / "_live_site"
# For Apache or LiteSpeed hosts (Hostinger, cPanel): data.json changes every minute, so no server, CDN or browser may
# keep an old copy; and the folder stays out of search. Hosts that ignore .htaccess are unaffected.
HTACCESS = """<IfModule mod_headers.c>
  Header set X-Robots-Tag "noindex"
  <Files "data.json">
    Header set Cache-Control "no-store"
  </Files>
</IfModule>
"""
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_IDLE = getattr(subprocess, "IDLE_PRIORITY_CLASS", 0)


def load_local():
    """config/live.local.yaml: upload, the command that copies the website folder to the site ({dir} is the folder,
    {name} the run); and url, the page's address (printed at the start). Empty if there is no such file."""
    path = ROOT / "config" / "live.local.yaml"
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.exists() else {}


def read_jsonl(path):
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:          # a line still being written
                pass
    return rows


def age_s(path):
    """Seconds since the log was last written: the page stops its motion when this gets long."""
    return round(time.time() - path.stat().st_mtime, 1) if path.exists() else None


def eta(steps_done, total, elapsed_s):
    if not steps_done or elapsed_s is None:
        return None, None
    per = elapsed_s / steps_done
    return per, per * max(total - steps_done, 0)


def run_start_ms(path, steps):
    """Wall-clock start of the run: from a logged time if the log has one, else the log's last write minus the
    last elapsed time (older logs recorded only elapsed seconds)."""
    if not steps:
        return path.stat().st_mtime * 1000 if path.exists() else None
    last = steps[-1]
    if "t" in last:
        return (last["t"] - last["elapsed_s"]) * 1000
    return (path.stat().st_mtime - last["elapsed_s"]) * 1000


def points(steps):
    """[step, loss, seconds since the start] per step; the page maps time to step with the third value."""
    return [[r["step"], r["loss"], r.get("elapsed_s")] for r in steps]


def memorise_data(path):
    rows = read_jsonl(path)
    meta = next((r["meta"] for r in rows if "meta" in r), {})
    steps = [r for r in rows if "step" in r]
    total = meta.get("steps", 0)
    last = steps[-1] if steps else {}
    per, left = eta(last.get("step"), total, last.get("elapsed_s"))
    done = any("f1" in r and all(r["f1"].get(p, 0) >= 0.9 for p in meta.get("must_pass", [])) for r in steps)
    result = None
    report = path.with_name("memorise.json")
    if report.exists() and path.exists() and report.stat().st_mtime >= path.stat().st_mtime - 5:
        result = json.loads(report.read_text(encoding="utf-8"))          # written at the end of this run
    return {"title": "Memorisation check" + (f" · control {meta['control']}" if meta.get("control") else "")
                     + (" · wide-target trial" if meta.get("wide_target") else ""),
            "kind": "memorise",
            "subtitle": f"The network practises on 8 short drum clips until it can pick out every kick, snare and "
                        f"closed hi-hat in them, for up to {total:,} steps."
                        + (" This is a control run." if meta.get("control") else "")
                        + (" Trial: it learns from a wider answer key that gives partial credit for near misses; "
                           "not the locked check." if meta.get("wide_target") else ""),
            "floor": meta.get("audio_blind_floor"), "total": total, "must_pass": meta.get("must_pass", []),
            "points": points(steps),
            "checks": [{"step": r["step"], "loss": r["loss"], "f1": r["f1"]} for r in steps if "f1" in r],
            "sec_per_step": per, "left_s": left, "done": done or (bool(steps) and last["step"] >= total) or bool(result),
            "result": result, "age_s": age_s(path), "run_start_ms": run_start_ms(path, steps)}


def train_data(run_dir):
    args = json.loads((run_dir / "args.json").read_text(encoding="utf-8")) if (run_dir / "args.json").exists() else {}
    log = run_dir / "log.jsonl"
    steps = [r for r in read_jsonl(log) if "step" in r]
    total = args.get("steps", 0)
    last = steps[-1] if steps else {}
    per, left = eta(last.get("step"), total, last.get("elapsed_s"))
    return {"title": f"Training run {run_dir.name}", "kind": "train",
            "subtitle": f"Training on the {args.get('set', '?')} set for {total:,} steps."
                        + (" This is a control run." if args.get("control") else ""),
            "floor": None, "total": total, "must_pass": [],
            "points": points(steps),
            "checks": [{"step": r["step"], "loss": r["val"]["loss"], "f1": r["val"]["f1"]} for r in steps if "val" in r],
            "sec_per_step": per, "left_s": left, "done": bool(steps) and last["step"] >= total, "result": None,
            "age_s": age_s(log), "run_start_ms": run_start_ms(log, steps)}


def memorise_pattern(name):
    """Command lines of the memorise.py run called name, for a chart watching it from outside: --name NAME, or
    for the default names no --name and the matching --control (none for "real")."""
    base = r"scripts[\\/]memorise\.py"
    named = base + r".*--name\s+" + re.escape(name) + r"(\s|$)"
    if name == "real":
        return rf"{named}|{base}(?!.*--(name|control)\s)"
    m = re.fullmatch(r"control-(\d+)", name)
    return rf"{named}|{base}(?!.*--name\s).*--control\s+{m[1]}(\s|$)" if m else named


def run_source(run=None, memorise="real"):
    """(data function, pattern matching the run's process, machine-log path) for a train.py run name, or the
    memorisation run called memorise."""
    if run:
        return (lambda: train_data(ROOT / "runs" / run), r"scripts[\\/]train\.py.*--run\s+" + re.escape(run) + r"(\s|$)",
                ROOT / "runs" / run / "machine_log.jsonl")
    folder = reports_dir() / "memorise" / memorise
    return lambda: memorise_data(folder / "log.jsonl"), memorise_pattern(memorise), folder / "machine_log.jsonl"


def site_payload(d, sampler, every_s, max_rows=1500):
    """Everything the website copy needs in one file: the run, the latest reading, and this run's readings
    (thinned to at most max_rows so the file stays small)."""
    start = (d["run_start_ms"] or time.time() * 1000) - 60_000
    if not d["points"]:
        start = min(start, time.time() * 1000 - 3_600_000)
    hist = sampler.since(start)
    k = max(1, -(-len(hist) // max_rows))
    thin = hist[::k] + ([hist[-1]] if hist and (len(hist) - 1) % k else [])
    m = sampler.latest
    return d | {"now_ms": time.time() * 1000, "update_every_s": every_s, "machine": m,
                "process_running": m["process_running"] if m else None, "history_cols": ROW, "history": thin,
                "sample_every_s": sampler.every * k}


def site_page():
    """The chart page, told to read data.json next to it instead of asking this server, and kept out of search."""
    page = PAGE.read_text(encoding="utf-8")
    page = page.replace('<meta name="viewport"', '<meta name="robots" content="noindex">\n<meta name="viewport"', 1)
    return page.replace("<script>\nconst $", '<script>\nwindow.__FEED__ = "data.json";\n</script>\n<script>\nconst $', 1)


def windows_path():
    """PATH as Windows has it now. A terminal opened before a tool was installed (rclone, say) still has the old one."""
    try:
        import winreg
    except ImportError:
        return os.environ.get("PATH", "")
    parts = []
    for hive, key in ((winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
                      (winreg.HKEY_CURRENT_USER, "Environment")):
        try:
            with winreg.OpenKey(hive, key) as k:
                parts.append(os.path.expandvars(winreg.QueryValueEx(k, "Path")[0]))
        except OSError:
            pass
    return ";".join(parts + [os.environ.get("PATH", "")])


class Publisher(threading.Thread):
    """Writes the website folder every every_s seconds and runs the upload command, at idle priority."""

    def __init__(self, name, every_s, upload, data, sampler, say=print):
        super().__init__(daemon=True)
        self.name, self.out_dir = name, SITE_ROOT
        self.every_s, self.upload = every_s, upload
        self.data, self.sampler, self.say = data, sampler, say
        self.lock, self.stopped = threading.Lock(), threading.Event()
        self.ok_before, self.page_mtime = None, None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / ".htaccess").write_text(HTACCESS, encoding="utf-8")

    def once(self, final=False):
        with self.lock:
            if self.stopped.is_set() and not final:    # the last update has gone out; nothing may follow it
                return
            if PAGE.stat().st_mtime != self.page_mtime:  # the page itself: at start, and again after any edit
                self.page_mtime = PAGE.stat().st_mtime
                (self.out_dir / "index.html").write_text(site_page(), encoding="utf-8")
            payload = site_payload(self.data(), self.sampler, self.every_s)
            if final:
                payload |= {"final": True, "process_running": False}   # the run is ending with this update
            tmp = self.out_dir / "data.json.tmp"
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp, self.out_dir / "data.json")   # never a half-written file
            if self.upload:
                ok, why = self._run(self.upload, self.out_dir)
                if ok is not None and ok != self.ok_before:         # say so only when it starts or stops working
                    self.say(f"{time.strftime('%H:%M:%S')} website upload {'working' if ok else 'FAILED: ' + why}")
                    self.ok_before = ok

    def _run(self, command, folder):
        """Run an upload command for folder: (True, "") if it worked, (False, why) if not, (None, "") if it timed out."""
        cmd = command.replace("{dir}", f'"{folder}"').replace("{name}", self.name)
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=max(30, self.every_s * 2),
                               env=os.environ | {"PATH": windows_path()}, creationflags=_NO_WINDOW | _IDLE)
        except subprocess.TimeoutExpired:
            self.say(f"{time.strftime('%H:%M:%S')} website upload took too long; will try again")
            return None, ""
        return r.returncode == 0, f"exit {r.returncode}: {(r.stderr or r.stdout).strip()[:300]}"

    def run(self):
        while not self.stopped.is_set():
            t = time.time()
            try:
                self.once()
            except Exception as e:                     # one bad moment must not end the updates for the whole run
                self.say(f"website update skipped: {e!r}")
            self.stopped.wait(max(5, self.every_s - (time.time() - t)))

    def finish(self):
        self.stopped.set()
        self.once(final=True)


class _Server(ThreadingHTTPServer):
    # Python's servers ask for SO_REUSEADDR, which on Windows lets a second server take a port already in use: two
    # charts would share 8765. Without it a taken port is refused, and the next run's chart moves on to 8766.
    allow_reuse_address = sys.platform != "win32"
    daemon_threads = True


def serve(port, data, sampler):
    """The page and its /data feed on localhost:port, from a background thread. False if the port is taken."""
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/data":
                d = data()
                since = float(parse_qs(url.query).get("since", ["0"])[0])
                if not since:                     # first load: this run's readings (the last hour if there is no run)
                    since = (d["run_start_ms"] or time.time() * 1000) - 60_000
                    since = min(since, time.time() * 1000 - 3_600_000) if not d["points"] else since
                m = sampler.latest
                d |= {"now_ms": time.time() * 1000, "machine": m, "process_running": m["process_running"] if m else None,
                      "history_cols": ROW, "history": sampler.since(since), "sample_every_s": sampler.every}
                body, ctype = json.dumps(d).encode(), "application/json"
            else:
                body, ctype = PAGE.read_bytes(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    try:
        server = _Server(("127.0.0.1", port), Handler)
    except OSError:
        return False
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return True


class LiveChart:
    """The live chart for one run: machine readings, the page on the first free port from `port` and, when publishing,
    the run's page on the website. memorise names a memorisation run, run a train.py run. own_process: the chart runs
    inside the run, so "this run" is this process. publish=None (memorise.py) publishes when config/live.local.yaml
    has an upload command. Call finish() when the run ends so the website keeps showing how it ended."""

    def __init__(self, memorise="real", run=None, port=8765, publish=None, upload=None, every_s=60, say=print,
                 own_process=False):
        self.name = run or memorise
        self.data, pattern, history = run_source(run, memorise)
        self.say, self.pub = say, None
        self.sampler = Sampler(pattern, history, own_process=own_process)
        self.port = next((p for p in range(port, port + 10) if serve(p, self.data, self.sampler)), None)
        say(f"live chart: http://localhost:{self.port}" if self.port else
            f"live chart: ports {port} to {port + 9} are all in use, so no local page this time")
        self.sampler.start()
        local = load_local()
        upload = upload or local.get("upload")
        if publish or (publish is None and upload):
            self.pub = Publisher(self.name, every_s, upload, self.data, self.sampler, say)
            self.pub.start()
            url = local.get("url")
            say(f"website: {url or self.pub.out_dir}"
                + (f", updated every {every_s} s" if upload else " (written only: no upload command)"))

    def finish(self):
        """Send the website one last update, marked final."""
        if self.pub:
            self.say("website: sending the last update")
            self.pub.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memorise", default="real", help="a memorisation run's name (default: real)")
    ap.add_argument("--run", help="a train.py run name, instead of a memorisation run")
    ap.add_argument("--port", type=int, default=8765, help="the first port to try")
    ap.add_argument("--publish", action="store_true", help="also update the run's page on the website")
    ap.add_argument("--publish-every", type=int, default=60, help="seconds between website updates")
    ap.add_argument("--upload", help="command that copies the website folder to your site ({dir} the folder, {name} "
                                     "the run); default: the one in config/live.local.yaml")
    args = ap.parse_args()
    chart = LiveChart(args.memorise, args.run, args.port, publish=args.publish, upload=args.upload,
                      every_s=args.publish_every)
    if not chart.port:
        sys.exit(1)
    print("Ctrl+C to stop")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
