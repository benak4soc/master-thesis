# ============================================================
# EVSE attack-detection dashboard for the Raspberry Pi 4.
#
# Stdlib HTTP server plus the NumPy LDA-LSTM already on the Pi.
# No Streamlit, Node, or CDN. Open http://<pi>:8080
#
# DoS and Cryptojacking replay recorded host windows through the
# real model. RomCom is not in the host test set: that button is a
# scripted RAT storyboard, and the page says so.
# shortcut: OCPP, talkers, and DoS packet rates are scenario playback,
# not a live capture. Mitigation buttons only change the demo log;
# they do not touch firewall rules.
# ============================================================

from __future__ import print_function

import argparse
import json
import os
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from rpi4_infer import default_deploy_dir, load_package, predict


CLASSES = ["Benign", "DoS", "RomCom", "Cryptojacking"]
ADVICE = {
    "Benign": "No mitigation needed.",
    "DoS": "Block the source IP and rate-limit new connections.",
    "RomCom": "Isolate the EVSE from the LAN and stop the unexpected outbound process.",
    "Cryptojacking": "Throttle or stop the miner and keep the charging session up.",
}
DOS_IPS = ["203.0.113.10", "203.0.113.23", "203.0.113.45", "198.51.100.8"]
ROMCOM_IP = "198.51.100.23"


def _push(buf, item, limit=40):
    buf.append(item)
    if len(buf) > limit:
        del buf[:-limit]


def _read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return None


def _cpu_sample():
    lines = _read_text("/proc/stat")
    if not lines:
        return None
    rows = []
    for line in lines.splitlines():
        if not line.startswith("cpu"):
            break
        parts = line.split()
        nums = [int(x) for x in parts[1:]]
        idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
        rows.append((parts[0], idle, sum(nums)))
    return rows


def _pct(prev, cur):
    idle = cur[0] - prev[0]
    total = cur[1] - prev[1]
    if total <= 0:
        return 0.0
    return max(0.0, min(100.0, 100.0 * (1.0 - idle / float(total))))


def _hex_ipv4(value):
    host = value.split(":")[0]
    if len(host) != 8:
        return None
    try:
        raw = bytes.fromhex(host)
    except ValueError:
        return None
    return ".".join(str(b) for b in raw[::-1])


def _live_talkers(limit=5):
    counts = {}
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        text = _read_text(path)
        if not text:
            continue
        for line in text.splitlines()[1:]:
            parts = line.split()
            if len(parts) < 4 or parts[3] != "01":
                continue
            ip = _hex_ipv4(parts[2])
            if not ip or ip.startswith("0.") or ip.startswith("127."):
                continue
            counts[ip] = counts.get(ip, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return [{"ip": ip, "share": count} for ip, count in ranked[:limit]]


def _syn_count():
    total = 0
    found = False
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        text = _read_text(path)
        if not text:
            continue
        found = True
        for line in text.splitlines()[1:]:
            parts = line.split()
            if len(parts) > 3 and parts[3] == "03":
                total += 1
    return total if found else None


def _tcp_inuse():
    text = _read_text("/proc/net/sockstat")
    if not text:
        return None
    for line in text.splitlines():
        if line.startswith("TCP:"):
            parts = line.split()
            if "inuse" in parts:
                return int(parts[parts.index("inuse") + 1])
    return None


def _net_totals():
    text = _read_text("/proc/net/dev")
    if not text:
        return None
    rx_b = rx_p = tx_b = tx_p = 0
    for line in text.splitlines():
        if ":" not in line:
            continue
        name, rest = line.split(":", 1)
        if name.strip() == "lo":
            continue
        parts = rest.split()
        rx_b += int(parts[0])
        rx_p += int(parts[1])
        tx_b += int(parts[8])
        tx_p += int(parts[9])
    return rx_b + tx_b, rx_p + tx_p


def _throttled():
    raw = _read_text("/sys/devices/platform/soc/soc:firmware/get_throttled")
    if raw is None:
        return None
    raw = raw.strip().lower().replace("0x", "")
    try:
        return int(raw, 16) != 0
    except ValueError:
        return None


def _mhz():
    raw = _read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    if not raw:
        return None
    try:
        return int(raw.strip()) / 1000.0
    except ValueError:
        return None


def _temp_c():
    raw = _read_text("/sys/class/thermal/thermal_zone0/temp")
    if not raw:
        return None
    try:
        return int(raw.strip()) / 1000.0
    except ValueError:
        return None


def _ram_pct():
    text = _read_text("/proc/meminfo")
    if not text:
        return None
    info = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        nums = val.strip().split()
        if nums and nums[0].isdigit():
            info[key] = int(nums[0])
    total = info.get("MemTotal")
    avail = info.get("MemAvailable")
    if not total or avail is None:
        return None
    return 100.0 * (total - avail) / float(total)


def _uptime_s():
    raw = _read_text("/proc/uptime")
    if not raw:
        return None
    try:
        return float(raw.split()[0])
    except (ValueError, IndexError):
        return None


def _load_test(deploy_dir):
    path = os.path.join(deploy_dir, "detection_report.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    metrics = report.get("test_metrics") or {}
    return {
        "n": metrics.get("n"),
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "roc_auc": metrics.get("roc_auc"),
        "confusion_matrix": metrics.get("confusion_matrix"),
        "per_family": report.get("per_family") or {},
    }


class Engine(object):
    def __init__(self):
        self.lock = threading.Lock()
        self.stop_loop = threading.Event()
        self._stress_stop = threading.Event()
        self._stress = None
        self.running = False
        self.scenario = "idle"
        self.mitigation = "off"
        self.sound = False
        self.energy = 0.0
        self.tick_i = 0
        self.alert_id = 0
        self._cpu_prev = None
        self._net_prev = None
        self._net_prev_t = None
        self.rng = random.Random(7)
        self.deploy = default_deploy_dir()
        self.error = None
        self.pack = None
        self.windows = {}
        try:
            self.pack = load_package(self.deploy)
            for name in ("benign", "dos", "cryptojacking"):
                self.windows[name] = np.load(
                    os.path.join(self.deploy, "samples", name + ".npy")
                )
        except Exception as exc:
            self.error = str(exc)
        meta = (self.pack or {}).get("meta") or {}
        self.state = {
            "status": "Idle",
            "attack_class": None,
            "confidence": None,
            "scenario": "idle",
            "running": False,
            "mitigation": "off",
            "sound": False,
            "model_name": "LDA-LSTM Pipeline C",
            "model_version": "tuned · hidden %s · %s params"
            % (meta.get("hidden_size", 16), meta.get("n_params", 1378)),
            "model_label": None,
            "model_p_benign": None,
            "model_p_attack": None,
            "model_note": self.error or "Press a scenario to replay a host window.",
            "latency_ms": None,
            "evse_state": "Idle",
            "uptime_s": _uptime_s(),
            "class_probs": {name: 0.0 for name in CLASSES},
            "prob_history": [],
            "log": [],
            "health": {
                "cpu": None,
                "cores": [],
                "temp_c": None,
                "ram_pct": None,
                "mhz": None,
                "throttled": None,
                "stress": False,
            },
            "health_history": [],
            "net": {
                "pps": 0.0,
                "bps": 0.0,
                "conns": None,
                "syn": None,
                "talkers": [],
                "live_pps": None,
                "live_bps": None,
                "mode": "live",
            },
            "net_history": [],
            "ocpp": {
                "rate": 0.0,
                "types": {},
                "voltage": None,
                "current": None,
                "power": None,
                "energy_kwh": None,
                "anomaly": "",
            },
            "host": {"processes": [], "beacons": "", "alerts": []},
            "summary": {
                "flows": 0,
                "attacks": 0,
                "false_alarms": 0,
                "by_class": {name: 0 for name in CLASSES},
            },
            "test": _load_test(self.deploy),
            "alert": None,
        }

    def snapshot(self):
        with self.lock:
            self.state["uptime_s"] = _uptime_s()
            return json.loads(json.dumps(self.state))

    def command(self, cmd, value=None):
        with self.lock:
            if cmd == "start":
                self.running = True
                if self.scenario == "idle":
                    self.scenario = "benign"
            elif cmd == "stop":
                self.running = False
                self.scenario = "idle"
                self._set_stress(False)
                self.state["status"] = "Idle"
                self.state["attack_class"] = None
                self.state["evse_state"] = "Idle"
                self.state["alert"] = None
                self.state["model_note"] = "Detector stopped."
            elif cmd == "scenario":
                if value not in ("benign", "dos", "romcom", "cryptojacking"):
                    return
                self.scenario = value
                self.running = True
                self.tick_i = 0
            elif cmd == "mitigation":
                if value in ("off", "block", "throttle", "isolate"):
                    self.mitigation = value
            elif cmd == "sound":
                self.sound = bool(value)
            self.state["running"] = self.running
            self.state["scenario"] = self.scenario
            self.state["mitigation"] = self.mitigation
            self.state["sound"] = self.sound
            self._set_stress(self.running and self.scenario == "cryptojacking")
            if self.running and self.scenario != "idle":
                self._infer_unlocked()
                self._telemetry_unlocked()
            self._health_unlocked()

    def loop(self):
        while not self.stop_loop.is_set():
            started = time.perf_counter()
            with self.lock:
                self._health_unlocked()
                if self.running and self.scenario != "idle":
                    self._infer_unlocked()
                    self._telemetry_unlocked()
            time.sleep(max(0.05, 1.0 - (time.perf_counter() - started)))

    def _set_stress(self, on):
        on = bool(on) and sys.platform.startswith("linux")
        self.state["health"]["stress"] = on
        if on:
            if self._stress is not None and self._stress.is_alive():
                return
            self._stress_stop.clear()

            def burn():
                while not self._stress_stop.is_set():
                    acc = 0
                    for i in range(250000):
                        acc += i * i

            self._stress = threading.Thread(target=burn, daemon=True)
            self._stress.start()
            return
        self._stress_stop.set()

    def _score(self, window):
        started = time.perf_counter()
        preds, probs, _ = predict(window, self.pack, threshold=0.5)
        latency = (time.perf_counter() - started) * 1000.0
        label = "Attack" if int(preds[0]) == 1 else "Benign"
        return label, float(probs[0, 0]), float(probs[0, 1]), latency

    def _infer_unlocked(self):
        if self.error or self.pack is None:
            self.state["model_note"] = self.error or "Model package missing."
            return
        self.tick_i += 1
        scenario = self.scenario
        if scenario == "romcom":
            label, p_benign, p_attack, latency = self._score(self.windows["benign"])
            probs = {
                "Benign": 0.04,
                "DoS": 0.03,
                "RomCom": round(self.rng.uniform(0.88, 0.94), 4),
                "Cryptojacking": 0.03,
            }
            probs["Benign"] = round(1.0 - probs["DoS"] - probs["RomCom"] - probs["Cryptojacking"], 4)
            status = "Attack"
            attack_class = "RomCom"
            note = (
                "Scripted RAT scenario. RomCom is not in the host test set, "
                "so this alert is not an LDA-LSTM class. "
                "The model scored a benign window as %s (attack %.0f%%)."
                % (label, 100.0 * p_attack)
            )
            confidence = probs["RomCom"]
        else:
            label, p_benign, p_attack, latency = self._score(self.windows[scenario])
            family = {"benign": "Benign", "dos": "DoS", "cryptojacking": "Cryptojacking"}[scenario]
            probs = {name: 0.0 for name in CLASSES}
            probs["Benign"] = round(p_benign, 4)
            if family == "Benign":
                status = "Attack" if label == "Attack" else "Benign"
                attack_class = None
                confidence = p_attack if status == "Attack" else p_benign
            else:
                probs[family] = round(p_attack, 4)
                status = "Attack" if label == "Attack" else "Benign"
                attack_class = family if status == "Attack" else None
                confidence = p_attack if status == "Attack" else p_benign
            note = "LDA-LSTM Benign vs Attack on a recorded %s host window." % scenario
        self.state["status"] = status
        self.state["attack_class"] = attack_class
        self.state["confidence"] = round(float(confidence), 4)
        self.state["class_probs"] = probs
        self.state["model_label"] = label
        self.state["model_p_benign"] = round(p_benign, 4)
        self.state["model_p_attack"] = round(p_attack, 4)
        self.state["model_note"] = note
        self.state["latency_ms"] = round(latency, 2)
        _push(self.state["prob_history"], probs)
        summary = self.state["summary"]
        summary["flows"] += 1
        log_cls = "Benign" if status != "Attack" else (attack_class or "Attack")
        if log_cls in summary["by_class"]:
            summary["by_class"][log_cls] += 1
        if status == "Attack":
            summary["attacks"] += 1
            title = "Attack detected · %s" % (attack_class or "Attack")
            previous = self.state["alert"]["title"] if self.state["alert"] else None
            if previous != title:
                self.alert_id += 1
            self.state["alert"] = {
                "id": self.alert_id,
                "title": title,
                "detail": ADVICE.get(attack_class or "DoS", ADVICE["DoS"]),
            }
        else:
            self.state["alert"] = None
        src = "—"
        if attack_class == "DoS":
            src = self.rng.choice(DOS_IPS)
        elif attack_class == "RomCom":
            src = ROMCOM_IP
        elif attack_class == "Cryptojacking":
            src = "local"
        action = "None"
        if status == "Attack":
            action = {
                "off": "Alert",
                "block": "Blocked IP",
                "throttle": "Throttled",
                "isolate": "Isolated EVSE",
            }[self.mitigation]
        _push(
            self.state["log"],
            {
                "ts": time.strftime("%H:%M:%S"),
                "cls": log_cls,
                "confidence": round(float(confidence), 4),
                "src": src,
                "action": action,
            },
            limit=80,
        )

    def _telemetry_unlocked(self):
        scenario = self.scenario
        if scenario == "dos":
            pps = self.rng.uniform(9000, 15000)
            bps = pps * self.rng.uniform(64, 140)
            voltage, current = 229.0, 0.0
            types = {"Heartbeat": 0, "MeterValues": 1, "StatusNotification": 0, "StartTransaction": 0}
            rate = 2.0
            anomaly = "Heartbeat timeout. MeterValues dropped and the session faulted."
            evse = "Faulted"
            processes = [
                {"name": "ocpp-client", "cpu": "1%", "note": "retries"},
                {"name": "python3 rpi4_dashboard.py", "cpu": "4%", "note": "detector"},
            ]
            beacons = "No periodic outbound beacon."
            alerts = ["Connection table grew with half-open sockets."]
            talkers = [
                {"ip": ip, "share": int(self.rng.uniform(800, 4000))}
                for ip in DOS_IPS
            ]
        elif scenario == "romcom":
            beacon = self.tick_i % 8 == 0
            pps = 36.0 if beacon else self.rng.uniform(4, 12)
            bps = pps * 180
            voltage = self.rng.uniform(229.5, 231.5)
            current = self.rng.uniform(15.6, 16.2)
            types = {"Heartbeat": 1, "MeterValues": 6, "StatusNotification": 1, "StartTransaction": 1}
            rate = 9.0
            anomaly = ""
            evse = "Charging"
            processes = [
                {"name": "ocpp-client", "cpu": "2%", "note": "normal"},
                {"name": "bash", "cpu": "1%", "note": "unexpected shell"},
                {"name": "curl " + ROMCOM_IP, "cpu": "0%", "note": "outbound"},
            ]
            beacons = "Periodic outbound burst to %s:443 about every 8 samples." % ROMCOM_IP
            alerts = ["New file /etc/cron.d/.update", "New outbound destination %s:443" % ROMCOM_IP]
            talkers = [{"ip": ROMCOM_IP, "share": 8 if beacon else 1}]
        elif scenario == "cryptojacking":
            pps = self.rng.uniform(15, 40)
            bps = pps * self.rng.uniform(70, 120)
            voltage = self.rng.uniform(228.5, 231.0)
            current = self.rng.uniform(16.8, 18.4)
            types = {"Heartbeat": 1, "MeterValues": 6, "StatusNotification": 1, "StartTransaction": 1}
            rate = 9.0
            anomaly = "Charger draw is steady, while host CPU stays high."
            evse = "Charging"
            cores = self.state["health"]["cores"] or []
            hot = max(cores) if cores else None
            processes = [
                {
                    "name": "crypto-miner",
                    "cpu": "high" if hot is None else "%.0f%%" % hot,
                    "note": "one core pegged",
                },
                {"name": "ocpp-client", "cpu": "2%", "note": "normal"},
                {"name": "python3 rpi4_dashboard.py", "cpu": "5%", "note": "detector"},
            ]
            beacons = "No command beacon. Pool-like outbound is not in the host window."
            alerts = ["Sustained high CPU on one core."]
            talkers = _live_talkers()
        else:
            pps = self.rng.uniform(18, 55)
            bps = pps * self.rng.uniform(80, 160)
            voltage = self.rng.uniform(229.0, 232.0)
            current = self.rng.uniform(15.4, 16.3)
            types = {"Heartbeat": 1, "MeterValues": 6, "StatusNotification": 1, "StartTransaction": 1}
            rate = 9.0
            anomaly = ""
            evse = "Charging"
            processes = [
                {"name": "ocpp-client", "cpu": "2%", "note": "normal"},
                {"name": "python3 rpi4_dashboard.py", "cpu": "4%", "note": "detector"},
            ]
            beacons = "No periodic outbound beacon."
            alerts = []
            talkers = _live_talkers()
        if self.mitigation == "isolate":
            evse = "Faulted"
        power = voltage * current / 1000.0
        self.energy += power * (1.0 / 3600.0)
        self.state["evse_state"] = evse
        self.state["ocpp"] = {
            "rate": rate,
            "types": types,
            "voltage": round(voltage, 1),
            "current": round(current, 2),
            "power": round(power, 2),
            "energy_kwh": round(self.energy, 4),
            "anomaly": anomaly,
        }
        self.state["host"] = {
            "processes": processes,
            "beacons": beacons,
            "alerts": alerts,
        }
        live_b, live_p = self._live_rates()
        mode = "live" if scenario == "idle" else "scenario"
        shown_pps, shown_bps = pps, bps
        if scenario == "idle" and live_p is not None:
            shown_pps, shown_bps = live_p, live_b
        self.state["net"] = {
            "pps": round(shown_pps, 1),
            "bps": round(shown_bps, 1),
            "conns": _tcp_inuse(),
            "syn": (420 + int(self.rng.uniform(0, 80))) if scenario == "dos" else _syn_count(),
            "talkers": talkers,
            "live_pps": None if live_p is None else round(live_p, 1),
            "live_bps": None if live_b is None else round(live_b, 1),
            "mode": mode,
        }
        _push(
            self.state["net_history"],
            {"pps": round(shown_pps, 1), "bps": round(shown_bps, 1)},
        )

    def _live_rates(self):
        totals = _net_totals()
        now = time.perf_counter()
        if totals is None:
            return None, None
        if self._net_prev is None:
            self._net_prev = totals
            self._net_prev_t = now
            return 0.0, 0.0
        dt = max(now - self._net_prev_t, 1e-3)
        bps = (totals[0] - self._net_prev[0]) / dt
        pps = (totals[1] - self._net_prev[1]) / dt
        self._net_prev = totals
        self._net_prev_t = now
        return max(0.0, bps), max(0.0, pps)

    def _health_unlocked(self):
        sample = _cpu_sample()
        cpu = None
        cores = []
        if sample is not None:
            if self._cpu_prev is not None and len(self._cpu_prev) == len(sample):
                cpu = _pct(self._cpu_prev[0][1:], sample[0][1:])
                for prev, cur in zip(self._cpu_prev[1:], sample[1:]):
                    cores.append(round(_pct(prev[1:], cur[1:]), 1))
            self._cpu_prev = sample
        health = {
            "cpu": None if cpu is None else round(cpu, 1),
            "cores": cores,
            "temp_c": None if _temp_c() is None else round(_temp_c(), 1),
            "ram_pct": None if _ram_pct() is None else round(_ram_pct(), 1),
            "mhz": _mhz(),
            "throttled": _throttled(),
            "stress": self.state["health"]["stress"],
        }
        self.state["health"] = health
        if cpu is not None:
            _push(
                self.state["health_history"],
                {
                    "cpu": health["cpu"],
                    "temp": health["temp_c"] if health["temp_c"] is not None else 0,
                },
            )


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EVSE attack detection</title>
<style>
  :root {
    --bg: #0c1218; --card: #15202b; --line: #2a3a4a; --text: #e7eef6;
    --muted: #93a4b8; --ok: #3dd68c; --bad: #ff6b6b; --dos: #ff8a3d;
    --rom: #c084fc; --crypto: #f5c16c; --info: #7eb6ff;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.4 "Segoe UI", system-ui, sans-serif; }
  header { display: flex; gap: 16px; align-items: center; padding: 12px 16px; background: #101820; border-bottom: 1px solid var(--line); flex-wrap: wrap; }
  h1 { font-size: 16px; margin: 0; font-weight: 650; }
  .sub { color: var(--muted); font-size: 12px; }
  .badge { font-size: 26px; font-weight: 750; padding: 8px 14px; border-radius: 12px; min-width: 280px; }
  .badge.ok { background: #123524; color: #b6ffd4; }
  .badge.bad { background: #3a1418; color: #ffd0d0; }
  .badge.idle { background: #1c2836; color: #d5dee8; }
  .chips { display: flex; gap: 8px; flex-wrap: wrap; }
  .chip { background: #1c2836; border: 1px solid var(--line); border-radius: 999px; padding: 4px 10px; color: var(--muted); }
  .chip strong { color: var(--text); font-weight: 650; }
  #banner { margin: 10px 12px 0; padding: 10px 12px; border-radius: 10px; background: #3a1418; color: #ffd7d7; display: none; }
  #banner strong { display: block; font-size: 16px; }
  .wrap { display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 10px; padding: 10px 12px 18px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 12px; min-width: 0; }
  .card h2 { margin: 0 0 4px; font-size: 12px; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
  .hint { color: var(--muted); font-size: 12px; min-height: 32px; }
  .conf { font-size: 28px; font-weight: 720; }
  .bars { display: grid; gap: 6px; margin: 8px 0; }
  .barline { display: grid; grid-template-columns: 110px 1fr 48px; gap: 8px; align-items: center; }
  .track { height: 10px; background: #0e1620; border-radius: 99px; overflow: hidden; }
  .fill { height: 100%; width: 0; }
  canvas { width: 100%; height: 120px; background: #0e1620; border-radius: 8px; display: block; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { text-align: left; padding: 5px 6px; border-bottom: 1px solid #243140; white-space: nowrap; }
  .scroll { max-height: 250px; overflow: auto; }
  tr.DoS td { background: #3a2412; }
  tr.RomCom td { background: #2c2040; }
  tr.Cryptojacking td { background: #3a3214; }
  tr.Benign td { background: #12261c; }
  .filters, .seg, .scenarios { display: flex; gap: 6px; flex-wrap: wrap; }
  button, .filters button { background: #1c2836; color: var(--text); border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; cursor: pointer; }
  button.on { outline: 2px solid var(--info); }
  button.scenario { font-weight: 650; }
  .metrics { display: flex; gap: 10px; flex-wrap: wrap; }
  .metric { background: #0e1620; border-radius: 8px; padding: 6px 8px; min-width: 72px; }
  .metric b { display: block; font-size: 16px; }
  .cores { display: grid; gap: 4px; margin-top: 6px; }
  .warn { color: #ffb4b4; }
  .oktext { color: var(--ok); }
  @media (max-width: 980px) { .wrap { grid-template-columns: 1fr; } .badge { min-width: 0; } }
  @media (prefers-reduced-motion: reduce) { * { animation: none !important; } }
</style>
</head>
<body>
<header>
  <div>
    <h1>EVSE security monitor</h1>
    <div class="sub">Raspberry Pi 4 · host LDA-LSTM</div>
  </div>
  <div id="badge" class="badge idle" aria-live="polite">Idle</div>
  <div class="chips">
    <div class="chip">EVSE <strong id="evse">Idle</strong></div>
    <div class="chip">Uptime <strong id="uptime">—</strong></div>
    <div class="chip">Model <strong id="model">LDA-LSTM</strong></div>
    <div class="chip">Mitigation <strong id="mit">off</strong></div>
  </div>
</header>
<div id="banner"></div>
<main class="wrap">
  <section class="card">
    <h2>Live detection</h2>
    <div class="conf" id="conf">—</div>
    <div class="bars" id="bars"></div>
    <canvas id="prob" width="640" height="120"></canvas>
    <p class="hint" id="latency"></p>
    <p class="hint" id="note"></p>
  </section>
  <section class="card">
    <h2>Attack timeline</h2>
    <div class="filters" id="filters"></div>
    <div class="scroll"><table><thead><tr><th>Time</th><th>Class</th><th>Confidence</th><th>Source IP</th><th>Action</th></tr></thead><tbody id="log"></tbody></table></div>
  </section>
  <section class="card">
    <h2>Pi 4 health</h2>
    <div class="metrics" id="healthnums"></div>
    <div class="cores" id="cores"></div>
    <canvas id="health" width="640" height="120"></canvas>
    <p class="hint" id="healthnote"></p>
  </section>
  <section class="card">
    <h2>Network</h2>
    <div class="metrics" id="netnums"></div>
    <canvas id="net" width="640" height="120"></canvas>
    <div id="talkers"></div>
    <p class="hint" id="netnote"></p>
  </section>
  <section class="card">
    <h2>EVSE / OCPP</h2>
    <div class="metrics" id="charge"></div>
    <div class="bars" id="ocpp"></div>
    <p class="hint" id="anomaly"></p>
  </section>
  <section class="card">
    <h2>Process and host</h2>
    <div id="procs"></div>
    <p class="hint" id="beacon"></p>
    <div id="falerts"></div>
  </section>
  <section class="card">
    <h2>Attack summary</h2>
    <div class="metrics" id="counts"></div>
    <canvas id="donut" width="640" height="140"></canvas>
    <div id="testset"></div>
  </section>
  <section class="card">
    <h2>Demo controls</h2>
    <div class="scenarios" id="scenarios"></div>
    <p></p>
    <div class="seg">
      <button type="button" data-cmd="start">Start</button>
      <button type="button" data-cmd="stop">Stop</button>
    </div>
    <p class="hint">Mitigation is recorded in the log only. It does not change firewall rules.</p>
    <div class="seg" id="mits"></div>
    <p><label><input type="checkbox" id="sound"> Alert sound</label></p>
    <p class="hint" id="advice"></p>
  </section>
</main>
<script>
const COLORS = {Benign:"#3dd68c", DoS:"#ff8a3d", RomCom:"#c084fc", Cryptojacking:"#f5c16c"};
const ORDER = ["Benign","DoS","RomCom","Cryptojacking"];
let filter = "All";
let lastAlert = 0;
let audioCtx = null;

function esc(s){return String(s == null ? "" : s).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}
function pct(x){return Math.round((Number(x) || 0) * 100);}
function uptime(s){
  if (s == null) return "—";
  s = Math.floor(s);
  return Math.floor(s/3600) + "h " + Math.floor((s%3600)/60) + "m";
}
function send(cmd, value){
  return fetch("/api/control", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({cmd: cmd, value: value})});
}
function fit(canvas){
  const r = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 300;
  const h = canvas.clientHeight || 120;
  if (canvas.width !== Math.floor(w*r) || canvas.height !== Math.floor(h*r)) {
    canvas.width = Math.floor(w*r);
    canvas.height = Math.floor(h*r);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(r,0,0,r,0,0);
  return {ctx: ctx, w: w, h: h};
}
function drawLines(canvas, series){
  const box = fit(canvas);
  const ctx = box.ctx;
  ctx.clearRect(0,0,box.w,box.h);
  ctx.strokeStyle = "#243140";
  ctx.strokeRect(0.5,0.5,box.w-1,box.h-1);
  series.forEach(function(line){
    const vals = line.values || [];
    if (!vals.length) return;
    let max = line.max;
    if (max == null) {
      max = 1;
      vals.forEach(function(v){ if (v > max) max = v; });
    }
    ctx.beginPath();
    ctx.strokeStyle = line.color;
    ctx.lineWidth = 2;
    vals.forEach(function(v, i){
      const x = vals.length === 1 ? box.w/2 : (i / (vals.length - 1)) * (box.w - 16) + 8;
      const y = box.h - 8 - (Math.max(0, v) / max) * (box.h - 20);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
}
function drawDonut(canvas, counts){
  const box = fit(canvas);
  const ctx = box.ctx;
  ctx.clearRect(0,0,box.w,box.h);
  const keys = ["DoS","RomCom","Cryptojacking"];
  const total = keys.reduce(function(s,k){ return s + (counts[k] || 0); }, 0);
  const cx = 70, cy = box.h/2, rad = 46;
  let angle = -Math.PI/2;
  if (!total) {
    ctx.strokeStyle = "#243140";
    ctx.lineWidth = 14;
    ctx.beginPath();
    ctx.arc(cx, cy, rad, 0, Math.PI*2);
    ctx.stroke();
  }
  keys.forEach(function(k){
    const n = counts[k] || 0;
    if (!n || !total) return;
    const slice = n / total * Math.PI * 2;
    ctx.beginPath();
    ctx.strokeStyle = COLORS[k];
    ctx.lineWidth = 14;
    ctx.arc(cx, cy, rad, angle, angle + slice);
    ctx.stroke();
    angle += slice;
  });
  ctx.fillStyle = "#e7eef6";
  ctx.font = "16px sans-serif";
  ctx.fillText(String(total), cx - 10, cy + 5);
  ctx.font = "13px sans-serif";
  keys.forEach(function(k, i){
    ctx.fillStyle = COLORS[k];
    ctx.fillText(k + " " + (counts[k] || 0), 140, 28 + i * 22);
  });
}
function meter(label, value, color, text){
  const width = Math.max(0, Math.min(100, value));
  const shown = text == null ? width.toFixed(0) + "%" : text;
  return '<div class="barline"><span>' + esc(label) + '</span><div class="track"><div class="fill" style="width:' + width + '%;background:' + color + '"></div></div><span>' + esc(shown) + '</span></div>';
}
function render(s){
  window.__state = s;
  const badge = document.getElementById("badge");
  const cls = s.attack_class;
  if (!s.running) {
    badge.className = "badge idle";
    badge.textContent = "Idle";
  } else if (s.status === "Attack") {
    badge.className = "badge bad";
    badge.textContent = "Attack detected · " + (cls || "Attack");
  } else {
    badge.className = "badge ok";
    badge.textContent = "Benign";
  }
  document.getElementById("evse").textContent = s.evse_state || "Idle";
  document.getElementById("uptime").textContent = uptime(s.uptime_s);
  document.getElementById("model").textContent = (s.model_name || "") + " " + (s.model_version || "");
  document.getElementById("mit").textContent = s.mitigation || "off";
  const banner = document.getElementById("banner");
  if (s.alert) {
    banner.style.display = "block";
    banner.innerHTML = "<strong>" + esc(s.alert.title) + "</strong>" + esc(s.alert.detail);
    if (s.sound && s.alert.id !== lastAlert) beep();
    lastAlert = s.alert.id;
  } else {
    banner.style.display = "none";
  }
  const conf = s.confidence == null ? "—" : (s.confidence * 100).toFixed(1) + "%";
  document.getElementById("conf").textContent = conf;
  document.getElementById("bars").innerHTML = ORDER.map(function(name){
    return meter(name, pct(s.class_probs[name]), COLORS[name]);
  }).join("");
  drawLines(document.getElementById("prob"), ORDER.map(function(name){
    return {color: COLORS[name], max: 1, values: (s.prob_history || []).map(function(row){ return row[name] || 0; })};
  }));
  document.getElementById("latency").textContent = s.latency_ms == null ? "Latency: —" : ("Latency: " + s.latency_ms.toFixed(2) + " ms / window");
  document.getElementById("note").textContent = s.model_note || "";
  const body = document.getElementById("log");
  const rows = (s.log || []).slice().reverse().filter(function(row){ return filter === "All" || row.cls === filter; });
  body.innerHTML = rows.map(function(row){
    return "<tr class=\"" + esc(row.cls) + "\"><td>" + esc(row.ts) + "</td><td>" + esc(row.cls) + "</td><td>" + (row.confidence*100).toFixed(1) + "%</td><td>" + esc(row.src) + "</td><td>" + esc(row.action) + "</td></tr>";
  }).join("");
  const h = s.health || {};
  document.getElementById("healthnums").innerHTML = [
    ["CPU", h.cpu == null ? "—" : h.cpu.toFixed(1) + "%"],
    ["Temp", h.temp_c == null ? "—" : h.temp_c.toFixed(1) + "°C"],
    ["RAM", h.ram_pct == null ? "—" : h.ram_pct.toFixed(1) + "%"],
    ["Clock", h.mhz == null ? "—" : Math.round(h.mhz) + " MHz"],
    ["Throttle", h.throttled == null ? "—" : (h.throttled ? "yes" : "no")]
  ].map(function(pair){ return '<div class="metric"><b>' + esc(pair[1]) + '</b>' + esc(pair[0]) + '</div>'; }).join("");
  document.getElementById("cores").innerHTML = (h.cores || []).map(function(v, i){ return meter("Core " + i, v, "#7eb6ff"); }).join("");
  drawLines(document.getElementById("health"), [
    {color:"#7eb6ff", max:100, values:(s.health_history||[]).map(function(row){ return row.cpu; })},
    {color:"#f5c16c", max:100, values:(s.health_history||[]).map(function(row){ return row.temp; })}
  ]);
  document.getElementById("healthnote").textContent = h.cpu == null
    ? "CPU, temperature, RAM, and throttling are read from the Pi."
    : (h.stress ? "Cryptojacking scenario is loading one core. Yellow line is temperature." : "Blue line is CPU %. Yellow line is temperature.");
  const n = s.net || {};
  document.getElementById("netnums").innerHTML = [
    ["Packets/s", n.pps == null ? "—" : Math.round(n.pps)],
    ["Bytes/s", n.bps == null ? "—" : Math.round(n.bps)],
    ["Connections", n.conns == null ? "—" : n.conns],
    ["SYN half-open", n.syn == null ? "—" : n.syn]
  ].map(function(pair){ return '<div class="metric"><b>' + esc(pair[1]) + '</b>' + esc(pair[0]) + '</div>'; }).join("");
  drawLines(document.getElementById("net"), [
    {color:"#ff8a3d", values:(s.net_history||[]).map(function(row){ return row.pps; })},
    {color:"#7eb6ff", values:(s.net_history||[]).map(function(row){ return row.bps; })}
  ]);
  document.getElementById("talkers").innerHTML = "<p class=\"hint\">Top talkers</p>" + (n.talkers || []).map(function(t){
    return '<div>' + esc(t.ip) + ' · ' + esc(t.share) + '</div>';
  }).join("");
  document.getElementById("netnote").textContent = n.mode === "scenario"
    ? "Orange is scenario packets/s, blue is scenario bytes/s. Live interface: " + (n.live_pps == null ? "n/a" : (Math.round(n.live_pps) + " pkt/s, " + Math.round(n.live_bps) + " B/s")) + "."
    : "Chart follows the live interface when a scenario is not running.";
  const o = s.ocpp || {};
  document.getElementById("charge").innerHTML = [
    ["Voltage", o.voltage == null ? "—" : o.voltage + " V"],
    ["Current", o.current == null ? "—" : o.current + " A"],
    ["Power", o.power == null ? "—" : o.power + " kW"],
    ["Energy", o.energy_kwh == null ? "—" : o.energy_kwh + " kWh"],
    ["OCPP", o.rate == null ? "—" : o.rate + "/min"]
  ].map(function(pair){ return '<div class="metric"><b>' + esc(pair[1]) + '</b>' + esc(pair[0]) + '</div>'; }).join("");
  const types = o.types || {};
  const maxType = Math.max.apply(null, [1].concat(Object.keys(types).map(function(k){ return types[k]; })));
  document.getElementById("ocpp").innerHTML = Object.keys(types).map(function(k){
    return meter(k, types[k] / maxType * 100, "#7eb6ff", String(types[k]));
  }).join("");
  const anomaly = document.getElementById("anomaly");
  anomaly.textContent = o.anomaly || "No meter or session anomaly in this scenario.";
  anomaly.className = o.anomaly ? "hint warn" : "hint";
  const host = s.host || {};
  document.getElementById("procs").innerHTML = "<table><thead><tr><th>Process</th><th>CPU</th><th>Note</th></tr></thead><tbody>" + (host.processes || []).map(function(p){
    return "<tr><td>" + esc(p.name) + "</td><td>" + esc(p.cpu) + "</td><td>" + esc(p.note) + "</td></tr>";
  }).join("") + "</tbody></table>";
  document.getElementById("beacon").textContent = host.beacons || "";
  document.getElementById("falerts").innerHTML = (host.alerts || []).map(function(a){ return '<div class="warn">' + esc(a) + '</div>'; }).join("");
  const sum = s.summary || {by_class:{}};
  document.getElementById("counts").innerHTML = [
    ["Flows", sum.flows || 0],
    ["Attacks", sum.attacks || 0],
    ["False alarms", sum.false_alarms || 0]
  ].map(function(pair){ return '<div class="metric"><b>' + esc(pair[1]) + '</b>' + esc(pair[0]) + '</div>'; }).join("");
  drawDonut(document.getElementById("donut"), sum.by_class || {});
  const test = s.test;
  const box = document.getElementById("testset");
  if (!test) {
    box.textContent = "";
  } else {
    const cm = test.confusion_matrix || [[0,0],[0,0]];
    const fam = test.per_family || {};
    box.innerHTML = '<p class="hint">Held-out host test, ' + esc(test.n) + ' windows. Binary model.</p>'
      + '<div class="metrics">'
      + metric("Accuracy", test.accuracy) + metric("Precision", test.precision) + metric("Recall", test.recall) + metric("F1", test.f1)
      + '</div><table><thead><tr><th></th><th>Pred benign</th><th>Pred attack</th></tr></thead><tbody>'
      + '<tr><td>Actual benign</td><td>' + cm[0][0] + '</td><td>' + cm[0][1] + '</td></tr>'
      + '<tr><td>Actual attack</td><td>' + cm[1][0] + '</td><td>' + cm[1][1] + '</td></tr></tbody></table>'
      + '<p class="hint">Family recall of the binary alert: '
      + Object.keys(fam).map(function(name){
          const row = fam[name];
          const rate = row.recall != null ? row.recall : row.rate;
          return name + " " + Math.round(rate * 100) + "%";
        }).join(" · ") + '</p>';
  }
  document.getElementById("advice").textContent = ADVICE_TEXT(s);
  document.querySelectorAll("button[data-value]").forEach(function(btn){
    const cmd = btn.getAttribute("data-cmd");
    const value = btn.getAttribute("data-value");
    const on = (cmd === "scenario" && s.scenario === value) || (cmd === "mitigation" && s.mitigation === value) || (cmd === "filter" && filter === value);
    btn.classList.toggle("on", on);
  });
  document.getElementById("sound").checked = !!s.sound;
}
function metric(label, value){
  const text = value == null ? "—" : (value * 100).toFixed(1) + "%";
  return '<div class="metric"><b>' + text + '</b>' + label + '</div>';
}
function ADVICE_TEXT(s){
  const key = s.attack_class || (s.status === "Benign" ? "Benign" : "Benign");
  const map = {Benign:"No mitigation needed.", DoS:"Block the source IP and rate-limit new connections.", RomCom:"Isolate the EVSE from the LAN and stop the unexpected outbound process.", Cryptojacking:"Throttle or stop the miner and keep the charging session up."};
  return map[key] || map.Benign;
}
function beep(){
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.frequency.value = 880;
    gain.gain.value = 0.05;
    osc.connect(gain); gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.15);
  } catch (err) {}
}
function boot(){
  const filters = document.getElementById("filters");
  ["All"].concat(ORDER).forEach(function(name){
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = name;
    btn.setAttribute("data-cmd", "filter");
    btn.setAttribute("data-value", name);
    btn.addEventListener("click", function(){ filter = name; if (window.__state) render(window.__state); });
    filters.appendChild(btn);
  });
  const scenarios = document.getElementById("scenarios");
  [["benign","Run Benign"],["dos","Simulate DoS"],["romcom","Simulate RomCom"],["cryptojacking","Simulate Cryptojacking"]].forEach(function(pair){
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "scenario";
    btn.textContent = pair[1];
    btn.setAttribute("data-cmd", "scenario");
    btn.setAttribute("data-value", pair[0]);
    btn.addEventListener("click", function(){ send("scenario", pair[0]); });
    scenarios.appendChild(btn);
  });
  const mits = document.getElementById("mits");
  [["off","Off"],["block","Block IP"],["throttle","Throttle"],["isolate","Isolate EVSE"]].forEach(function(pair){
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = pair[1];
    btn.setAttribute("data-cmd", "mitigation");
    btn.setAttribute("data-value", pair[0]);
    btn.addEventListener("click", function(){ send("mitigation", pair[0]); });
    mits.appendChild(btn);
  });
  document.querySelector("button[data-cmd=start]").addEventListener("click", function(){ send("start"); });
  document.querySelector("button[data-cmd=stop]").addEventListener("click", function(){ send("stop"); });
  document.getElementById("sound").addEventListener("change", function(ev){ send("sound", ev.target.checked); });
  setInterval(poll, 1000);
  poll();
}
function poll(){
  fetch("/api/state").then(function(r){ return r.json(); }).then(render).catch(function(){});
}
boot();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _send(self, code, body, content_type):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
            return
        if path == "/api/state":
            self._send(200, json.dumps(ENGINE.snapshot()), "application/json")
            return
        self._send(404, "Not found", "text/plain; charset=utf-8")

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/control":
            self._send(404, "Not found", "text/plain; charset=utf-8")
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except ValueError:
            self._send(400, json.dumps({"error": "bad json"}), "application/json")
            return
        ENGINE.command(payload.get("cmd"), payload.get("value"))
        self._send(200, json.dumps(ENGINE.snapshot()), "application/json")


ENGINE = None


def self_check():
    engine = Engine()
    if engine.error:
        raise SystemExit(engine.error)
    engine.command("scenario", "dos")
    if engine.state["model_label"] != "Attack" or engine.state["attack_class"] != "DoS":
        raise SystemExit("DoS window was not detected: %s" % engine.state["model_label"])
    if engine.state["latency_ms"] is None:
        raise SystemExit("latency missing")
    engine.command("scenario", "benign")
    if engine.state["status"] != "Benign":
        raise SystemExit("benign window flagged as attack")
    engine.command("scenario", "cryptojacking")
    if engine.state["attack_class"] != "Cryptojacking":
        raise SystemExit("cryptojacking window was not detected")
    engine.command("scenario", "romcom")
    if engine.state["attack_class"] != "RomCom":
        raise SystemExit("RomCom scenario did not arm")
    if "not in the host" not in engine.state["model_note"]:
        raise SystemExit("RomCom note missing")
    engine.command("mitigation", "block")
    if engine.state["log"][-1]["action"] != "Blocked IP":
        raise SystemExit("mitigation was not recorded")
    engine.command("stop")
    if engine.state["running"] or engine.state["health"]["stress"]:
        raise SystemExit("stop did not halt the detector")
    print("self-check ok")


def serve(host, port):
    global ENGINE
    ENGINE = Engine()
    if ENGINE.error:
        print(ENGINE.error)
    worker = threading.Thread(target=ENGINE.loop, daemon=True)
    worker.start()
    httpd = ThreadingHTTPServer((host, port), Handler)
    print("EVSE dashboard http://%s:%s" % (host, port), flush=True)
    httpd.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="EVSE detection dashboard")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
