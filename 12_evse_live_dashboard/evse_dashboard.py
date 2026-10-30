# EVSE attack-detection dashboard for the Jetson Nano demo.
# Pipeline C scores Benign vs Attack. DoS / RomCom / Cryptojacking
# are the scenario being replayed, drawn on top of that live score.
#
#   python evse_dashboard.py
#   python evse_dashboard.py --self-check

from __future__ import print_function

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INFER_DIR = os.path.normpath(os.path.join(HERE, "..", "11_jetson_nano_demo"))
sys.path.insert(0, INFER_DIR)

from jetson_infer import (  # noqa: E402
    build_backend,
    default_deploy_dir,
    load_meta,
    load_preprocess,
    preprocess_windows,
    softmax,
)

CLASSES = ("Benign", "DoS", "RomCom", "Cryptojacking")
SCENARIOS = ("benign", "dos", "romcom", "cryptojacking")
MITIGATIONS = ("off", "block", "throttle", "isolate")

ADVICE = {
    "Benign": "No mitigation. The charging session can continue.",
    "DoS": "Block the flooding source and rate-limit new SYNs. Resume charging once OCPP heartbeats return.",
    "RomCom": "Isolate the EVSE from the WAN, stop the new process, and check the OCPP binary before the next session.",
    "Cryptojacking": "Throttle or kill the miner, confirm CPU and temperature fall, then allow charging again.",
    "Attack": "The model raised Attack with no scenario family. Check CPU, new processes, and outbound connections.",
}

# shortcut: host, network, and OCPP numbers are scenario replay, not /proc or a packet capture. Read the Nano's own counters when the demo must show live hardware.


def _clip(value, lo, hi):
    return max(lo, min(hi, value))


def _mix(a, b, t):
    return a * (1.0 - t) + b * t


class Engine(object):
    def __init__(self, deploy_dir):
        self.deploy_dir = deploy_dir
        self.lock = threading.Lock()
        self.running = False
        self.scenario = "benign"
        self.mitigation = "off"
        self.recover = 0.0
        self.replay = None
        self.replay_i = 0
        self.started = time.time()
        self.rng = np.random.default_rng()
        self.kwh = 0.42
        self.flows = 0
        self.attacks = 0
        self.false_alarms = 0
        self.tp = self.fp = self.tn = self.fn = 0
        self.counts = {name: 0 for name in CLASSES}
        self.history = {
            "Benign": [],
            "DoS": [],
            "RomCom": [],
            "Cryptojacking": [],
            "cpu": [],
            "temp": [],
            "pps": [],
            "bps": [],
            "power": [],
        }
        self.log = []
        self.file_alerts = []
        self.error = None
        self.backend_name = None
        self._infer = None
        self.meta = None
        self.prep = None
        self.benign_row = None
        self.public = None
        self._load()
        self.public = self._blank()

    def _load(self):
        self.meta = load_meta(self.deploy_dir)
        self.prep = load_preprocess(self.deploy_dir)
        if self.prep["mode"] != "numpy":
            raise RuntimeError("preprocess_numpy.npz is required in the deploy folder")
        self.backend_name, self._infer = build_backend(self.deploy_dir, self.meta)
        t = int(self.meta["n_timesteps"])
        f = int(self.meta["n_features_raw"])
        probe = np.random.default_rng(1).normal(size=(t, f)).astype(np.float32)
        lda = preprocess_windows(probe, self.prep)[0, 0]
        pinv = np.linalg.pinv(self.prep["scalings"])
        scaled = self.prep["xbar"] + np.dot(lda, pinv)
        self.benign_row = (scaled * self.prep["scale"] + self.prep["mean"]).astype(np.float32)

    def _blank(self):
        test = self.meta["checkpoint_metrics"]["test"]
        n_params = self.meta["checkpoint_metrics"].get("n_params", "")
        return {
            "running": False,
            "scenario": self.scenario,
            "mitigation": self.mitigation,
            "badge": "Standby",
            "badge_kind": "idle",
            "evse_state": "Idle",
            "uptime_s": 0,
            "model_name": "Pipeline C LDA-LSTM",
            "model_version": "ONNX · %s params" % n_params,
            "model_label": "—",
            "confidence": None,
            "probs": {name: 0.0 for name in CLASSES},
            "latency_ms": None,
            "backend": self.backend_name,
            "cpu": 0,
            "cores": [0, 0, 0, 0],
            "temp_c": 0,
            "ram": 0,
            "throttle": False,
            "clock_ghz": 1.43,
            "pps": 0,
            "bps": 0,
            "conns": 0,
            "syn": 0,
            "talkers": [],
            "ocpp_rate": 0,
            "ocpp": {},
            "voltage": 0,
            "current": 0,
            "power_kw": 0,
            "energy_kwh": self.kwh,
            "processes": [],
            "outbound": [],
            "beacon": False,
            "beacon_s": None,
            "file_alerts": [],
            "advice": ADVICE["Benign"],
            "action": "none",
            "counts": dict(self.counts),
            "flows": 0,
            "attacks": 0,
            "false_alarms": 0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "precision": None,
            "recall": None,
            "test_accuracy": test.get("accuracy"),
            "test_f1": test.get("f1"),
            "test_auc": test.get("roc_auc"),
            "history": {key: [] for key in self.history},
            "log": [],
            "banner": "",
            "error": self.error,
        }

    def control(self, body):
        cmd = body.get("cmd")
        value = body.get("value")
        with self.lock:
            if cmd == "start":
                self.running = True
            elif cmd == "stop":
                self.running = False
            elif cmd == "scenario":
                if value not in SCENARIOS:
                    raise ValueError("unknown scenario %s" % value)
                self.scenario = value
                self.replay = None
                self.recover = 0.0
                self.file_alerts = []
                self.running = True
            elif cmd == "mitigation":
                if value not in MITIGATIONS:
                    raise ValueError("unknown mitigation %s" % value)
                self.mitigation = value
                self.recover = 0.0
            elif cmd == "npy":
                raw = np.load(value)
                if raw.ndim == 2:
                    raw = raw[None, ...]
                if raw.ndim != 3:
                    raise ValueError("expected (T,F) or (B,T,F)")
                self.replay = raw.astype(np.float32)
                self.replay_i = 0
                self.scenario = "benign"
                self.running = True
            else:
                raise ValueError("unknown cmd %s" % cmd)

    def loop(self):
        while True:
            if self.running:
                self.step()
            time.sleep(1.0)

    def step(self):
        with self.lock:
            self._step()

    def _window(self):
        t = int(self.meta["n_timesteps"])
        f = int(self.meta["n_features_raw"])
        if self.replay is not None:
            raw = self.replay[self.replay_i % len(self.replay)]
            self.replay_i += 1
            return raw.astype(np.float32)
        if self.scenario == "benign":
            raw = np.tile(self.benign_row, (t, 1))
            raw = raw + self.rng.normal(0, 0.02, size=raw.shape).astype(np.float32)
            return raw
        raw = np.full((t, f), -6, np.float32)
        raw = raw + self.rng.normal(0, 0.15, size=raw.shape).astype(np.float32)
        return raw

    def _advance_recover(self):
        engaged = (
            self.mitigation == "isolate"
            or (self.mitigation == "block" and self.scenario == "dos")
            or (self.mitigation == "throttle" and self.scenario == "cryptojacking")
        )
        if engaged:
            self.recover = min(1.0, self.recover + 0.34)
        else:
            self.recover = max(0.0, self.recover - 0.34)

    def _scene(self):
        profiles = {
            "benign": dict(cpu=14, temp=46, ram=31, pps=42, bps=18000, conns=6, syn=0, evse="Charging", volt=230.0, amp=16.0, throttle=False, clock=1.43),
            "dos": dict(cpu=52, temp=63, ram=44, pps=14500, bps=9200000, conns=460, syn=390, evse="Faulted", volt=207.0, amp=2.4, throttle=False, clock=1.43),
            "romcom": dict(cpu=23, temp=52, ram=38, pps=58, bps=42000, conns=11, syn=1, evse="Charging", volt=229.0, amp=15.5, throttle=False, clock=1.43),
            "cryptojacking": dict(cpu=98, temp=83, ram=74, pps=28, bps=11000, conns=8, syn=0, evse="Charging", volt=230.0, amp=15.7, throttle=True, clock=0.92),
        }
        benign = profiles["benign"]
        src = profiles[self.scenario]
        t = self.recover
        scene = {}
        for key in ("cpu", "temp", "ram", "pps", "bps", "conns", "syn", "volt", "amp", "clock"):
            scene[key] = _mix(src[key], benign[key] if self.mitigation != "isolate" else 0, t)
        if self.mitigation == "isolate":
            scene["pps"] = _mix(src["pps"], 2, t)
            scene["bps"] = _mix(src["bps"], 400, t)
            scene["conns"] = _mix(src["conns"], 1, t)
            scene["syn"] = _mix(src["syn"], 0, t)
            scene["cpu"] = _mix(src["cpu"], 8, t)
            scene["temp"] = _mix(src["temp"], 44, t)
            scene["evse"] = "Faulted"
            scene["throttle"] = False
            scene["clock"] = 1.43
        else:
            scene["evse"] = "Charging" if t > 0.67 and self.scenario == "dos" else src["evse"]
            scene["throttle"] = bool(src["throttle"] and t < 0.5)
            scene["clock"] = src["clock"] if scene["throttle"] else 1.43
        if self.mitigation == "throttle" and self.scenario == "cryptojacking":
            scene["cpu"] = _mix(src["cpu"], 36, t)
            scene["temp"] = _mix(src["temp"], 55, t)
            scene["throttle"] = t < 0.5
            scene["clock"] = 0.92 if scene["throttle"] else 1.43
        j = self.rng.uniform
        scene["cpu"] = _clip(scene["cpu"] * j(0.97, 1.03), 1, 99)
        scene["temp"] = _clip(scene["temp"] + j(-0.6, 0.6), 35, 95)
        scene["ram"] = _clip(scene["ram"] * j(0.98, 1.02), 5, 95)
        scene["pps"] = max(0, scene["pps"] * j(0.94, 1.06))
        scene["bps"] = max(0, scene["bps"] * j(0.94, 1.06))
        scene["conns"] = int(max(0, round(scene["conns"] * j(0.95, 1.05))))
        scene["syn"] = int(max(0, round(scene["syn"] * j(0.95, 1.05))))
        scene["volt"] = scene["volt"] + j(-0.8, 0.8)
        scene["amp"] = max(0.0, scene["amp"] + j(-0.15, 0.15))
        return scene

    def _cores(self, cpu, scenario):
        if scenario == "cryptojacking" and self.mitigation != "throttle":
            return [round(_clip(cpu + self.rng.uniform(-1.5, 1.5), 1, 99), 1) for _ in range(4)]
        if scenario == "dos":
            base = [cpu + 18, cpu, cpu - 8, cpu - 6]
        elif scenario == "benign":
            base = [cpu - 4, cpu + 3, cpu - 2, cpu + 1]
        else:
            base = [cpu + 6, cpu - 3, cpu, cpu - 2]
        return [round(_clip(v, 1, 99), 1) for v in base]

    def _side_channels(self, scene):
        scenario = self.scenario
        isolated = self.mitigation == "isolate" and self.recover > 0.5
        throttled = self.mitigation == "throttle" and scenario == "cryptojacking" and self.recover > 0.5
        if scenario == "dos":
            talkers = [
                {"ip": "203.0.113.10", "pps": int(scene["pps"] * 0.78)},
                {"ip": "203.0.113.11", "pps": int(scene["pps"] * 0.14)},
                {"ip": "198.51.100.8", "pps": int(scene["pps"] * 0.08)},
            ]
            src = "203.0.113.10"
        elif scenario == "romcom":
            talkers = [
                {"ip": "192.0.2.20", "pps": 12},
                {"ip": "198.51.100.23", "pps": 6 if not isolated else 0},
            ]
            src = "198.51.100.23"
        elif scenario == "cryptojacking":
            talkers = [
                {"ip": "192.0.2.20", "pps": 8},
                {"ip": "203.0.113.50", "pps": 4 if not isolated else 0},
            ]
            src = "203.0.113.50"
        else:
            talkers = [
                {"ip": "192.0.2.20", "pps": int(max(1, scene["pps"] * 0.7))},
                {"ip": "192.0.2.1", "pps": int(max(1, scene["pps"] * 0.3))},
            ]
            src = "192.0.2.20"

        if scenario == "dos":
            ocpp = {"Heartbeat": 0, "MeterValues": 1, "StatusNotification": 1, "StartTransaction": 0, "StopTransaction": 1}
        elif scenario == "romcom":
            ocpp = {"Heartbeat": 1, "MeterValues": 3, "StatusNotification": 1, "StartTransaction": 0, "StopTransaction": 0}
        elif scenario == "cryptojacking":
            ocpp = {"Heartbeat": 1, "MeterValues": 1, "StatusNotification": 0, "StartTransaction": 0, "StopTransaction": 0}
        else:
            ocpp = {"Heartbeat": 1, "MeterValues": 4, "StatusNotification": 1, "StartTransaction": 0, "StopTransaction": 0}
        if isolated:
            ocpp = {key: 0 for key in ocpp}

        if scenario == "romcom" and not isolated:
            processes = [
                {"name": "ocpp-client", "cpu": 4.0, "flag": ""},
                {"name": ".systemd-update", "cpu": 11.0, "flag": "suspicious"},
                {"name": "sshd", "cpu": 0.4, "flag": ""},
            ]
            outbound = [{"dst": "198.51.100.23", "port": 8443, "note": "new, periodic"}]
            beacon = True
        elif scenario == "cryptojacking":
            miner_cpu = 8.0 if throttled or isolated else 94.0
            processes = [
                {"name": "ocpp-client", "cpu": 3.0, "flag": ""},
                {"name": "xmrig", "cpu": miner_cpu, "flag": "throttled" if throttled else "suspicious"},
                {"name": "sshd", "cpu": 0.3, "flag": ""},
            ]
            outbound = [] if isolated else [{"dst": "203.0.113.50", "port": 3333, "note": "mining pool"}]
            beacon = False
        elif scenario == "dos":
            processes = [
                {"name": "ocpp-client", "cpu": 6.0, "flag": ""},
                {"name": "ksoftirqd/0", "cpu": 22.0, "flag": ""},
                {"name": "sshd", "cpu": 0.2, "flag": ""},
            ]
            outbound = [{"dst": "192.0.2.20", "port": 443, "note": "CSMS"}]
            beacon = False
        else:
            processes = [
                {"name": "ocpp-client", "cpu": 3.2, "flag": ""},
                {"name": "sshd", "cpu": 0.2, "flag": ""},
                {"name": "python3", "cpu": 1.1, "flag": ""},
            ]
            outbound = [{"dst": "192.0.2.20", "port": 443, "note": "CSMS"}]
            beacon = False

        if scenario == "romcom" and not isolated and len(self.file_alerts) < 3:
            stamps = (
                "/tmp/.font-unix created",
                "/usr/local/bin/ocpp-helper replaced",
                "/etc/cron.d/evse-update added",
            )
            nxt = stamps[len(self.file_alerts)]
            if self.flows % 3 == 1:
                self.file_alerts.append(nxt)
        if scenario != "romcom":
            self.file_alerts = []

        if scenario == "dos" and self.mitigation == "block" and self.recover > 0.5:
            action = "blocked 203.0.113.10"
        elif isolated:
            action = "EVSE isolated"
        elif throttled:
            action = "host CPU throttled"
        elif scenario != "benign":
            action = "alert only"
        else:
            action = "none"
        return talkers, src, ocpp, processes, outbound, beacon, action

    def _step(self):
        self._advance_recover()
        raw = self._window()
        started = time.perf_counter()
        x_lda = preprocess_windows(raw, self.prep)
        probs_bin = softmax(self._infer(x_lda))[0]
        latency_ms = (time.perf_counter() - started) * 1000.0
        p_benign = float(probs_bin[0])
        p_attack = float(probs_bin[1])
        model_attack = p_attack >= p_benign
        model_label = "Attack" if model_attack else "Benign"

        family = {
            "dos": "DoS",
            "romcom": "RomCom",
            "cryptojacking": "Cryptojacking",
        }.get(self.scenario)
        if self.replay is not None:
            badge = model_label
        elif model_attack and family:
            badge = family
        elif model_attack:
            badge = "Attack"
        else:
            badge = "Benign"

        probs = {name: 0.0 for name in CLASSES}
        probs["Benign"] = p_benign
        if badge in ("DoS", "RomCom", "Cryptojacking"):
            probs[badge] = p_attack
        elif badge == "Attack":
            probs["DoS"] = p_attack
        else:
            share = p_attack / 3.0
            probs["DoS"] = share
            probs["RomCom"] = share
            probs["Cryptojacking"] = share

        scene = self._scene()
        talkers, src, ocpp, processes, outbound, beacon, action = self._side_channels(scene)
        power_kw = scene["volt"] * scene["amp"] / 1000.0
        self.kwh += power_kw / 3600.0
        scenario_attack = self.scenario != "benign" and self.replay is None
        self.flows += 1
        if model_attack:
            self.attacks += 1
        if model_attack and not scenario_attack:
            self.false_alarms += 1
            self.fp += 1
        elif model_attack and scenario_attack:
            self.tp += 1
        elif (not model_attack) and scenario_attack:
            self.fn += 1
        else:
            self.tn += 1
        if badge in self.counts:
            self.counts[badge] += 1

        for name in CLASSES:
            self.history[name].append(round(probs[name], 4))
            self.history[name] = self.history[name][-60:]
        self.history["cpu"].append(round(scene["cpu"], 2))
        self.history["temp"].append(round(scene["temp"], 2))
        self.history["pps"].append(round(scene["pps"], 1))
        self.history["bps"].append(round(scene["bps"], 1))
        self.history["power"].append(round(power_kw, 3))
        for key in ("cpu", "temp", "pps", "bps", "power"):
            self.history[key] = self.history[key][-60:]

        row = {
            "ts": time.strftime("%H:%M:%S"),
            "cls": badge,
            "conf": round(p_attack if model_attack else p_benign, 4),
            "src": src,
            "action": action,
        }
        self.log.insert(0, row)
        self.log = self.log[:240]

        precision = (self.tp / float(self.tp + self.fp)) if (self.tp + self.fp) else None
        recall = (self.tp / float(self.tp + self.fn)) if (self.tp + self.fn) else None
        advice_key = badge if badge in ADVICE else "Attack"
        state = self._blank()
        state.update({
            "running": True,
            "scenario": self.scenario,
            "mitigation": self.mitigation,
            "badge": badge,
            "badge_kind": "ok" if badge == "Benign" else "bad",
            "evse_state": scene["evse"],
            "uptime_s": int(time.time() - self.started),
            "model_label": model_label,
            "confidence": row["conf"],
            "probs": {key: round(val, 4) for key, val in probs.items()},
            "latency_ms": round(latency_ms, 3),
            "cpu": round(scene["cpu"], 1),
            "cores": self._cores(scene["cpu"], self.scenario),
            "temp_c": round(scene["temp"], 1),
            "ram": round(scene["ram"], 1),
            "throttle": bool(scene["throttle"]),
            "clock_ghz": scene["clock"],
            "pps": round(scene["pps"], 1),
            "bps": round(scene["bps"], 1),
            "conns": scene["conns"],
            "syn": scene["syn"],
            "talkers": talkers,
            "ocpp_rate": int(sum(ocpp.values())),
            "ocpp": ocpp,
            "voltage": round(scene["volt"], 1),
            "current": round(scene["amp"], 2),
            "power_kw": round(power_kw, 3),
            "energy_kwh": round(self.kwh, 4),
            "processes": processes,
            "outbound": outbound,
            "beacon": beacon,
            "beacon_s": 4.0 if beacon else None,
            "file_alerts": list(self.file_alerts),
            "advice": ADVICE[advice_key],
            "action": action,
            "counts": dict(self.counts),
            "flows": self.flows,
            "attacks": self.attacks,
            "false_alarms": self.false_alarms,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision": None if precision is None else round(precision, 3),
            "recall": None if recall is None else round(recall, 3),
            "history": {key: list(val) for key, val in self.history.items()},
            "log": list(self.log),
            "banner": "" if badge == "Benign" else ADVICE[advice_key],
        })
        self.public = state

    def snapshot(self):
        with self.lock:
            data = json.loads(json.dumps(self.public))
            data["uptime_s"] = int(time.time() - self.started)
            data["running"] = self.running and data.get("badge") != "Standby"
            if not self.running and self.public.get("badge") != "Standby":
                data["running"] = False
            return data


def self_check(deploy_dir):
    engine = Engine(deploy_dir)
    engine.control({"cmd": "scenario", "value": "benign"})
    engine.step()
    snap = engine.snapshot()
    assert snap["badge"] == "Benign", snap["badge"]
    assert snap["model_label"] == "Benign", snap["model_label"]
    assert abs(sum(snap["probs"].values()) - 1.0) < 0.02
    assert snap["latency_ms"] > 0
    assert snap["evse_state"] == "Charging"

    engine.control({"cmd": "scenario", "value": "dos"})
    engine.step()
    snap = engine.snapshot()
    assert snap["badge"] == "DoS", snap["badge"]
    assert snap["model_label"] == "Attack"
    assert snap["pps"] > 5000, snap["pps"]
    assert snap["syn"] > 100, snap["syn"]

    engine.control({"cmd": "mitigation", "value": "block"})
    for _ in range(4):
        engine.step()
    snap = engine.snapshot()
    assert snap["pps"] < 800, snap["pps"]
    assert snap["evse_state"] == "Charging", snap["evse_state"]
    assert snap["action"] == "blocked 203.0.113.10"

    engine.control({"cmd": "scenario", "value": "cryptojacking"})
    engine.control({"cmd": "mitigation", "value": "off"})
    engine.step()
    snap = engine.snapshot()
    assert snap["badge"] == "Cryptojacking", snap["badge"]
    assert snap["cpu"] > 90, snap["cpu"]
    assert snap["throttle"] is True

    engine.control({"cmd": "mitigation", "value": "throttle"})
    for _ in range(4):
        engine.step()
    snap = engine.snapshot()
    assert snap["cpu"] < 50, snap["cpu"]

    engine.control({"cmd": "scenario", "value": "romcom"})
    engine.control({"cmd": "mitigation", "value": "off"})
    engine.flows = 1
    engine.step()
    snap = engine.snapshot()
    assert snap["badge"] == "RomCom", snap["badge"]
    assert snap["beacon"] is True
    assert snap["file_alerts"], snap["file_alerts"]
    assert any(p["flag"] == "suspicious" for p in snap["processes"])
    print("self-check ok")
    print("benign/dos/crypto/romcom badges match the model score")


def serve(engine, host, port, open_browser):
    root = os.path.join(HERE, "dashboard.html")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, content_type):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _json(self, payload, code=200):
            self._send(code, json.dumps(payload), "application/json")

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/dashboard.html"):
                with open(root, "rb") as handle:
                    self._send(200, handle.read(), "text/html; charset=utf-8")
                return
            if path == "/api/state":
                self._json(engine.snapshot())
                return
            self._send(404, "not found", "text/plain")

        def do_POST(self):
            if self.path.split("?", 1)[0] != "/api/control":
                self._send(404, "not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            try:
                engine.control(json.loads(raw))
            except Exception as exc:
                self._json({"ok": False, "error": str(exc)}, 400)
                return
            self._json({"ok": True})

        def log_message(self, fmt, *args):
            return

    ThreadingHTTPServer.allow_reuse_address = True
    httpd = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=engine.loop, daemon=True).start()
    url = "http://127.0.0.1:%d/" % port
    print("EVSE dashboard  %s" % url, flush=True)
    print("model           %s (%s)" % (engine.backend_name, engine.deploy_dir), flush=True)
    if open_browser:
        webbrowser.open(url)
    httpd.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="EVSE Jetson attack-detection dashboard")
    parser.add_argument("--deploy-dir", default=default_deploy_dir())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not os.path.isdir(args.deploy_dir):
        raise SystemExit("Deploy dir not found: %s" % args.deploy_dir)
    if args.self_check:
        self_check(args.deploy_dir)
        return
    engine = Engine(args.deploy_dir)
    serve(engine, args.host, args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
