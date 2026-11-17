#!/usr/bin/env python3
"""Host-only graphical tuning of the competition MPC; never a ROS node.

Uses the official IROS Unity image unchanged. Weights are read by MPC at node
startup, so Apply restarts only the controller container, not the Unity player.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import yaml

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "tools/tuner/profiles/competition_gui.yaml"
STATE = Path.home() / ".local/share/sdu-apex/competition-tuner"
ACTIVE = STATE / "active.yaml"
SESSION = os.environ.get("SDU_APEX_TUNER_SESSION", uuid.uuid4().hex)
CONTROLLER = "sdu_apex_autodrive_competition"
UNITY = "sdu_apex_sim_iros_gui"
DEFAULT_IMAGE = "sdu-apex-autodrive:competition"
UNITY_IMAGE = "autodriveecosystem/autodrive_roboracer_sim@sha256:749fbef07942109d18497cbcf6ffe9452e06ae487f2bfc915e92440bc4b4663d"
LABEL = "sdu.apex.tuner.session"
DEST_CONFIG = "/workspace/install/share/f1tenth_mpc/config/mpc_competition.yaml"

# Display only objective and optional speed-envelope parameters. All other
# launch/solver parameters stay exactly as in the selected profile file.
GROUPS = [
    ("Speed and corner approach", [
        ("weight_u", "Under-target speed weight"),
        ("weight_u_overspeed", "Overspeed weight"),
        ("overspeed_curvature_gain_m", "Overspeed curvature gain (m)"),
        ("underspeed_curvature_gain_m", "Underspeed curvature gain (m)"),
        ("curvature_speed_envelope_lateral_accel_mps2", "Curvature speed cap: lateral accel (0 = off)"),
        ("curvature_speed_envelope_braking_mps2", "Pre-corner braking limit (m/s²)"),
        ("curvature_speed_envelope_acceleration_mps2", "Corner-exit acceleration limit (m/s²)"),
    ]),
    ("Heading and steering", [
        ("weight_e_y", "Lateral error weight"),
        ("weight_e_psi", "Heading error weight"),
        ("lateral_curvature_gain_m", "Lateral curvature gain (m)"),
        ("heading_curvature_gain_m", "Heading curvature gain (m)"),
        ("weight_steering_command", "Steering command penalty"),
        ("weight_steering_target", "Reference steering penalty"),
        ("weight_steering_slew", "Steering correction slew penalty"),
    ]),
    ("Dynamics, grip and smoothness", [
        ("weight_grip_loss", "Rear-slip / grip loss penalty"),
        ("grip_loss_tolerance", "Grip-loss tolerance"),
        ("weight_v", "Lateral velocity penalty"),
        ("weight_r", "Yaw-rate tracking penalty"),
        ("weight_throttle_command", "Throttle command penalty"),
        ("weight_throttle_slew", "Throttle slew penalty"),
        ("terminal_multiplier", "Terminal cost multiplier"),
    ]),
]
ALL_KEYS = [key for _, group in GROUPS for key, _ in group]
OPTIONAL_DEFAULTS = {"curvature_speed_envelope_braking_mps2": 5.0,
                     "curvature_speed_envelope_acceleration_mps2": 5.0}


def run(*args: str, check: bool = True, timeout: float = 9) -> subprocess.CompletedProcess:
    result = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(f"{' '.join(args[:3])} failed: {result.stderr.strip() or result.stdout.strip()}")
    return result


def owned(name: str) -> bool:
    result = run("docker", "inspect", "--format", '{{index .Config.Labels "'+LABEL+'"}}', name, check=False)
    return result.returncode == 0 and result.stdout.strip() == SESSION


def exists(name: str) -> bool:
    return run("docker", "container", "inspect", name, check=False).returncode == 0


def running(name: str) -> bool:
    r = run("docker", "inspect", "--format", "{{.State.Running}}", name, check=False)
    return r.returncode == 0 and r.stdout.strip() == "true"


def bridge_listening() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", 4567)) == 0


def read_profile(path: Path) -> tuple[str, dict]:
    content = path.read_text(encoding="utf-8")
    data = yaml.safe_load(content)
    if not isinstance(data, dict) or not isinstance(data.get("/**"), dict):
        raise ValueError("Expected ROS 2 /**: ros__parameters: YAML format")
    params = data["/**"].get("ros__parameters")
    if not isinstance(params, dict):
        raise ValueError("Missing ros__parameters")
    for key, default in OPTIONAL_DEFAULTS.items():
        params.setdefault(key, default)
    for key in ALL_KEYS:
        if key not in params or not isinstance(params[key], (float, int)) or isinstance(params[key], bool):
            raise ValueError(f"Missing or nonnumeric parameter: {key}")
    return content, params


def validate_values(values: dict[str, float]) -> None:
    for key, value in values.items():
        if not math.isfinite(value) or value < 0 or value > 100_000_000:
            raise ValueError(f"{key}: enter a finite, nonnegative value")
    for name in ("overspeed_curvature_gain_m", "underspeed_curvature_gain_m"):
        if values[name] > 10:
            raise ValueError(f"{name} must not exceed 10")
    for name in ("heading_curvature_gain_m", "lateral_curvature_gain_m"):
        if values[name] > 10:
            raise ValueError(f"{name} must not exceed 10 for this tuning tool")
    if values["curvature_speed_envelope_lateral_accel_mps2"] > 40:
        raise ValueError("Curvature speed cap lateral acceleration must not exceed 40")
    for name in ("curvature_speed_envelope_braking_mps2", "curvature_speed_envelope_acceleration_mps2"):
        if not 0 < values[name] <= 30:
            raise ValueError(f"{name} must be greater than zero and no more than 30")


def yaml_double(value: float) -> str:
    """Keep ROS2 declare_parameter<double> compatible, including 0.0/100.0."""
    encoded = repr(float(value))
    if "e" in encoded or "E" in encoded:
        head, exponent = re.split("[eE]", encoded, maxsplit=1)
        if "." not in head:
            head += ".0"
        encoded = head + "e" + exponent
    if not isinstance(yaml.safe_load(encoded), float):
        raise ValueError("Cannot represent ROS 2 floating-point parameter")
    return encoded


def render_profile(original: str, values: dict[str, float]) -> str:
    validate_values(values)
    s = original
    for name, value in values.items():
        pattern = rf"(?m)^(\s*{re.escape(name)}:\s*)[^#\n]*(\s*(?:\#[^\n]*)?)$"
        s, n = re.subn(pattern, lambda m: m.group(1) + yaml_double(value) + m.group(2), s)
        if n == 0 and name in OPTIONAL_DEFAULTS:
            anchor = "    curvature_speed_envelope_lateral_accel_mps2:"
            lines = s.splitlines(keepends=True)
            idxs = [i for i, line in enumerate(lines) if line.startswith(anchor)]
            if len(idxs) != 1:
                raise ValueError("Missing curvature envelope anchor")
            lines.insert(idxs[0] + 1, f"    {name}: {yaml_double(value)}\n")
            s = "".join(lines)
            n = 1
        if n != 1:
            raise ValueError(f"Expected exactly one YAML entry for {name}; found {n}")
    parsed = yaml.safe_load(s)["/**"]["ros__parameters"]
    for name, value in values.items():
        if type(parsed[name]) is not float or parsed[name] != float(value):
            raise ValueError(f"{name}: YAML must preserve the ROS 2 double type")
    return s


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + SESSION + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


class CompetitionSession:
    def __init__(self, image: str, width: int = 1280, height: int = 720):
        self.image = image
        self.width, self.height = width, height
        self.controller_process: subprocess.Popen | None = None
        self.lock = threading.RLock()

    def _stop_owned(self, name: str) -> None:
        if owned(name):
            run("docker", "stop", "--time", "3", name, check=False, timeout=15)
        elif exists(name):
            raise RuntimeError(f"Refusing to stop external container: {name}")

    def _start_controller(self) -> None:
        if exists(CONTROLLER):
            raise RuntimeError("Competition controller container already exists; refusing to take it over")
        if bridge_listening():
            raise RuntimeError("TCP 4567 occupied by another process; cannot start a second bridge")
        STATE.mkdir(parents=True, exist_ok=True)
        log = (STATE / "controller_launcher.log").open("a", encoding="utf-8")
        env = os.environ.copy()
        env.update({"SDU_APEX_IMAGE": self.image,
                    "SDU_APEX_CONTAINER": CONTROLLER,
                    "SDU_APEX_MPC_CONFIG_OVERRIDE": str(ACTIVE),
                    "SDU_APEX_TUNER_SESSION": SESSION})
        try:
            self.controller_process = subprocess.Popen(
                [str(ROOT / "tools/run_competition.sh")],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
        finally:
            log.close()
        deadline = time.monotonic() + 65
        while time.monotonic() < deadline:
            if self.controller_process.poll() is not None:
                raise RuntimeError("Controller exited during startup; see " + str(STATE / "controller_launcher.log"))
            if owned(CONTROLLER) and bridge_listening():
                return
            time.sleep(.35)
        raise RuntimeError("Controller did not listen on TCP port 4567")

    def _start_unity(self) -> None:
        if exists(UNITY):
            raise RuntimeError("Unity container already exists; refusing to take it over")
        display = os.getenv("DISPLAY")
        authority = os.getenv("XAUTHORITY")
        if not display or not authority or not Path(authority).is_file():
            raise RuntimeError("Log into your graphical desktop first (DISPLAY/XAUTHORITY missing)")
        authority = str(Path(authority).resolve())
        run("docker", "image", "inspect", UNITY_IMAGE)
        # Keep the official player and same fixed graphics isolation as previous tests.
        cmd = ("if command -v taskset >/dev/null; then "
               "taskset --pid --cpu-list 12-15 $$ >/dev/null || exit 1; fi; "
               "exec './AutoDRIVE Simulator.x86_64' -ip 127.0.0.1 "
               "-port 4567 -logFile /dev/null "
               f"-screen-width {self.width} -screen-height {self.height} -screen-fullscreen 0")
        run("docker", "run", "-d", "--rm", "--log-driver=none",
            "--name", UNITY, "--label", f"{LABEL}={SESSION}",
            "--network=host", "--ipc=host", "--privileged", "--gpus", "all",
            "--ulimit", "core=0", "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display",
            "-e", f"DISPLAY={display}", "-e", f"XAUTHORITY={authority}",
            "-v", f"{authority}:{authority}:ro", "-v", "/tmp/.X11-unix:/tmp/.X11-unix:rw",
            "-w", "/home/autodrive_simulator", "--entrypoint", "/bin/bash",
            UNITY_IMAGE, "-lc", cmd, timeout=20)
        time.sleep(1.2)
        if not owned(UNITY) or not running(UNITY):
            raise RuntimeError("Graphical Unity exited during startup")

    def start(self) -> str:
        with self.lock:
            if exists(UNITY) or exists(CONTROLLER):
                raise RuntimeError("A simulator or controller is already running. Stop it before starting this tool")
            if run("powerprofilesctl", "get", check=False).stdout.strip() != "performance":
                raise RuntimeError("Switch host to Performance power profile before starting")
            if not ACTIVE.is_file():
                raise RuntimeError("No active MPC YAML exists")
            try:
                self._start_controller()
                self._start_unity()
            except Exception:
                self.stop()
                raise
            return "Unity started. Press Connect. AMCL and MPC start after fresh sensors."

    def restart_controller(self) -> str:
        with self.lock:
            if not owned(UNITY) or not running(UNITY):
                raise RuntimeError("Unity is not running under this tuning session. Use Start Session")
            self._stop_owned(CONTROLLER)
            if self.controller_process is not None:
                try:self.controller_process.wait(timeout=9)
                except subprocess.TimeoutExpired:self.controller_process.terminate()
            for _ in range(45):
                if not exists(CONTROLLER) and not bridge_listening():break
                time.sleep(.2)
            self._start_controller()
            if not owned(UNITY) or not running(UNITY):
                raise RuntimeError("Unity exited during controller restart")
            return "New MPC parameters loaded. Unity stayed open. Press Connect again if needed, then Unity Restart."

    def ros_node_state(self) -> str:
        """Check actual MPC component presence, not just its container."""
        if not owned(CONTROLLER) or not running(CONTROLLER):
            return "Controller container stopped"
        command = ("source /opt/ros/humble/setup.bash; "
                   "source /workspace/install/setup.bash; "
                   "timeout 4s ros2 node list --no-daemon")
        result = run("docker", "exec", CONTROLLER, "bash", "-lc", command,
                     check=False, timeout=7)
        if result.returncode:
            return "ROS node inspection unavailable"
        nodes = set(result.stdout.splitlines())
        if "/mpc_controller_node" in nodes:
            socket_state = run("ss", "-tin", "( sport = :4567 )",
                               check=False, timeout=3)
            if socket_state.returncode != 0 or "ESTAB" not in socket_state.stdout:
                return "MPC LOADED — Unity not connected; press Connect"
            ages = re.findall(r"\blastrcv:(\d+)", socket_state.stdout)
            if ages and min(map(int, ages)) > 750:
                return ("MPC LOADED — Unity data stopped for " +
                        f"{min(map(int, ages))/1000:.1f}s; press Connect / Restart")
            return "MPC LOADED — Unity data arriving"
        if "/competition_ready_sensors" in nodes:
            return "Waiting for Unity Connect / fresh sensors"
        if "/competition_ready_pose" in nodes:
            return "Waiting for valid AMCL map pose"
        if "/controller_container" in nodes:
            return "ERROR: MPC component NOT LOADED (check floating-point YAML types)"
        return "Waiting for controller startup"

    def stop(self) -> str:
        with self.lock:
            # Stop Unity first, to avoid leaving an uncontrolled moving vehicle.
            try:self._stop_owned(UNITY)
            finally:self._stop_owned(CONTROLLER)
            if self.controller_process is not None:
                try:self.controller_process.wait(timeout=6)
                except subprocess.TimeoutExpired:
                    self.controller_process.terminate()
                self.controller_process = None
            return "Stopped tuner-owned Unity and controller containers."


def smoke_test(image: str) -> None:
    print("Starting test controller + graphical Unity, without pressing Connect", flush=True)
    content, params = read_profile(ACTIVE if ACTIVE.exists() else BASE)
    write_atomic(ACTIVE, render_profile(content, {k: float(params[k]) for k in ALL_KEYS}))
    session = CompetitionSession(image)
    try:
        print(session.start(), flush=True)
        unity_id = run("docker", "inspect", "-f", "{{.Id}}", UNITY).stdout.strip()
        controller_id = run("docker", "inspect", "-f", "{{.Id}}", CONTROLLER).stdout.strip()
        print("BEFORE", unity_id[:14], controller_id[:14], flush=True)
        print(session.restart_controller(), flush=True)
        assert run("docker", "inspect", "-f", "{{.Id}}", UNITY).stdout.strip() == unity_id
        assert run("docker", "inspect", "-f", "{{.Id}}", CONTROLLER).stdout.strip() != controller_id
        print("PASS: Unity container ID preserved; controller ID changed", flush=True)
    finally:
        print(session.stop(), flush=True)
        assert not exists(UNITY) and not exists(CONTROLLER), "Failed to clean up test containers"
        print("PASS: both tuner-owned containers removed", flush=True)

class CompetitionTunerGUI:
    def __init__(self, root: tk.Tk, image: str):
        self.root = root
        self.session = CompetitionSession(image)
        self.queue: queue.Queue = queue.Queue()
        self.busy = False
        self.close_requested = False
        self.node_check_pending = False
        self.last_node_state = ""
        self.form: dict[str, tk.StringVar] = {}
        self.profile_template = ""
        self.loaded_profile_name = ""
        self.status = tk.StringVar(value="Ready to start. Unity connects manually.")
        root.title("SDU Apex | Competition MPC Tuner")
        root.geometry("1030x790")
        root.minsize(820, 650)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._make_ui()
        self._load(ACTIVE if ACTIVE.exists() else BASE)
        self._process_queue()
        self._status_poll()

    def _make_ui(self) -> None:
        shell = ttk.Frame(self.root, padding=12)
        shell.pack(fill="both", expand=True)
        ttk.Label(shell, text="Competition MPC | Live Test Tuner",
                  font=("Sans", 17, "bold")).pack(anchor="w", pady=(0, 5))
        ttk.Label(shell, text="Edit values, then Apply & Restart MPC. The graphical Unity simulator stays open.",
                  font=("Sans", 10)).pack(anchor="w")
        ttk.Label(shell, text="Unity Restart resets the vehicle, NOT the MPC weights. Reconnect after a controller restart if needed.",
                  foreground="#a55b12", wraplength=970).pack(anchor="w", pady=(2, 2))
        ttk.Label(shell, text="Only apply new settings when the vehicle is stationary or after a crash; restarting control while moving is unsafe.",
                  foreground="#a52a2a", wraplength=970).pack(anchor="w", pady=(0, 8))
        ttk.Label(shell, text=f"MPC image: {self.session.image}     |     Unity: official IROS graphical player (1280 × 720)",
                  foreground="#596477").pack(anchor="w", pady=(0, 6))

        controls = ttk.Frame(shell)
        controls.pack(fill="x", pady=(0, 7))
        self.buttons = []
        for label, action in [
            ("Start Unity + MPC", self._start),
            ("Apply & Restart MPC", self._apply),
            ("Stop Both", self._stop),
            ("Load Profile…", self._pick_load),
            ("Save Profile As…", self._pick_save),
            ("Restore Test Defaults", self._restore_defaults),
        ]:
            btn = ttk.Button(controls, text=label, command=action)
            btn.pack(side="left", padx=(0, 7), pady=2)
            self.buttons.append(btn)
        ttk.Label(shell, textvariable=self.status, wraplength=960).pack(anchor="w", pady=(0, 5))
        self.profile_path_label = ttk.Label(shell, text="", foreground="#596477")
        self.profile_path_label.pack(anchor="w", pady=(0, 7))

        area = ttk.Frame(shell)
        area.pack(fill="both", expand=True)
        canvas = tk.Canvas(area, highlightthickness=0)
        scroll = ttk.Scrollbar(area, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        frame = ttk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=frame, anchor="nw")
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(window_id, width=e.width))
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))
        for group_name, fields in GROUPS:
            group = ttk.LabelFrame(frame, text=group_name, padding=9)
            group.pack(fill="x", padx=(0, 6), pady=(0, 9))
            for i, (key, label) in enumerate(fields):
                col = i % 2
                row = i // 2
                cell = ttk.Frame(group)
                cell.grid(row=row, column=col, sticky="ew", padx=8, pady=5)
                group.columnconfigure(col, weight=1)
                ttk.Label(cell, text=label, width=37).pack(side="left")
                var = tk.StringVar()
                ttk.Entry(cell, textvariable=var, width=11).pack(side="left", padx=(4, 3))
                self.form[key] = var
        ttk.Label(shell, text="Session events (host only; no diagnostic load inside the competition image)",
                  font=("Sans", 10, "bold")).pack(anchor="w", pady=(9, 3))
        panel = ttk.Frame(shell)
        panel.pack(fill="both")
        self.events = tk.Text(panel, height=6, wrap="word", state="disabled", font=("Monospace", 9))
        self.events.pack(side="left", fill="both", expand=True)
        ttk.Scrollbar(panel, command=self.events.yview).pack(side="right", fill="y")
        self._log("Ready. Default test profile: heading=2500, heading curvature=0.5, steering slew=500, overspeed=50, overspeed curvature=1.25. Loaded active profile may differ.")

    def _log(self, message: str) -> None:
        self.events.configure(state="normal")
        self.events.insert("end", f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.events.see("end")
        self.events.configure(state="disabled")

    def _load(self, path: Path) -> None:
        original, params = read_profile(path)
        self.profile_template = original
        self.loaded_profile_name = str(path)
        for key in ALL_KEYS:
            self.form[key].set(f"{float(params[key]):g}")
        self.profile_path_label.configure(text="Loaded: " + self.loaded_profile_name)
        self._log("Loaded profile: " + str(path))

    def _values(self) -> dict[str, float]:
        values = {}
        for key in ALL_KEYS:
            try:values[key] = float(self.form[key].get().strip())
            except ValueError:raise ValueError(f"{key}: enter a number") from None
        validate_values(values)
        return values

    def _render(self) -> str:
        return render_profile(self.profile_template, self._values())

    def _save_active(self) -> None:
        profile = self._render()
        write_atomic(ACTIVE, profile)
        snap = STATE / "history" / (time.strftime("%Y%m%d_%H%M%S") + "_" + SESSION[:6] + ".yaml")
        write_atomic(snap, profile)
        self._log(f"Saved active YAML and snapshot: {snap.name}")

    def _run_worker(self, description: str, action, close_after=False) -> None:
        if self.busy:
            return
        self.busy = True
        for btn in self.buttons:btn.configure(state="disabled")
        self.status.set(description + " …")
        self._log(description)
        def worker():
            try:self.queue.put((True, action(), close_after))
            except Exception as error:self.queue.put((False, f"{type(error).__name__}: {error}", close_after))
        threading.Thread(target=worker, daemon=True).start()

    def _process_queue(self) -> None:
        try:
            while True:
                item = self.queue.get_nowait()
                if item[0] == "health":
                    self.node_check_pending = False
                    message = item[1]
                    if message != self.last_node_state:
                        self.last_node_state = message
                        self._log("ROS readiness: " + message)
                        if message.startswith("ERROR:") or "Unity data stopped" in message or "Unity not connected" in message:
                            self.status.set(message)
                        elif message == "MPC LOADED — Unity data arriving":
                            self.status.set("MPC component loaded; Unity TCP stream is live. Reset vehicle in Unity if required.")
                    continue
                success, message, close_after = item
                self.busy = False
                for btn in self.buttons:btn.configure(state="normal")
                self.status.set(message)
                self._log(("OK: " if success else "ERROR: ") + message)
                if close_after:
                    self.root.destroy()
                    return
                if self.close_requested:
                    self._run_worker("Stopping before window closes", self.session.stop, close_after=True)
        except queue.Empty:
            pass
        self.root.after(200, self._process_queue)

    def _start(self) -> None:
        try:self._save_active()
        except Exception as e:
            messagebox.showerror("Invalid parameters", str(e), parent=self.root)
            return
        self._run_worker("Starting competition MPC and graphical Unity", self.session.start)

    def _apply(self) -> None:
        try:self._render()
        except Exception as e:
            messagebox.showerror("Invalid parameters", str(e), parent=self.root)
            return
        if not messagebox.askyesno(
            "Apply new MPC parameters",
            "Is the vehicle stationary or stopped after a collision?\n\n"
            "The MPC controller will restart, but Unity will stay open. "
            "Use Connect again if needed, then Unity Restart.\n\n"
            "Continuing while the vehicle is moving can cause a collision.",
            parent=self.root):
            return
        try:self._save_active()
        except Exception as e:
            messagebox.showerror("Could not save configuration", str(e), parent=self.root)
            return
        self._run_worker("Reloading MPC parameters (keeping Unity open)", self.session.restart_controller)

    def _stop(self) -> None:
        self._run_worker("Stopping tuner-owned containers", self.session.stop)

    def _pick_load(self) -> None:
        filename = filedialog.askopenfilename(parent=self.root, title="Load MPC tuning profile",
                                              initialdir=str(STATE), filetypes=[("YAML profiles", "*.yaml *.yml")])
        if filename:
            try:self._load(Path(filename))
            except Exception as e:messagebox.showerror("Invalid profile", str(e), parent=self.root)

    def _pick_save(self) -> None:
        filename = filedialog.asksaveasfilename(parent=self.root, title="Save MPC tuning profile",
                                                initialdir=str(STATE), defaultextension=".yaml",
                                                filetypes=[("YAML profiles", "*.yaml *.yml")])
        if filename:
            try:
                write_atomic(Path(filename), self._render())
                self._log("Saved profile: " + filename + " (not applied until restart)")
            except Exception as e:messagebox.showerror("Cannot save profile", str(e), parent=self.root)

    def _restore_defaults(self) -> None:
        self._load(BASE)
        self._log("Test defaults restored in editor; click Apply to use them")

    def _status_poll(self) -> None:
        if not self.busy and not self.close_requested:
            try:
                unity = running(UNITY) and owned(UNITY)
                controller = running(CONTROLLER) and owned(CONTROLLER)
                if controller and unity and not self.node_check_pending:
                    self.node_check_pending = True
                    def check_nodes():
                        try:
                            self.queue.put(("health", self.session.ros_node_state()))
                        except Exception as error:
                            self.queue.put(("health", "ROS status error: " + str(error)))
                    threading.Thread(target=check_nodes, daemon=True).start()
                status = ("[MPC + UNITY STREAM]" if unity and controller and
                          self.last_node_state == "MPC LOADED — Unity data arriving" else
                          "[UNITY NOT STREAMING]" if unity and controller and
                          ("Unity not connected" in self.last_node_state or
                           "Unity data stopped" in self.last_node_state) else
                          "[WAITING FOR MPC]" if unity and controller else
                          "[UNITY ONLY]" if unity else "[STOPPED]")
                self.root.title("SDU Apex | Competition MPC Tuner  " + status)
                if controller and not unity:
                    self._log("Warning: Unity exited; stopping its controller")
                    self._run_worker("Unity exited; stopping MPC", self.session.stop)
            except Exception as error:
                self._log("Status inspection failed: " + str(error))
        self.root.after(3500, self._status_poll)

    def _on_close(self) -> None:
        if self.close_requested:return
        self.close_requested = True
        self._log("Window closing; stopping only this session's containers")
        if not self.busy:
            self._run_worker("Closing and cleaning up", self.session.stop, close_after=True)
        else:
            self.status.set("Current operation finishing; automatic cleanup will follow")


def main() -> None:
    parser = argparse.ArgumentParser(description="Host-only graphical competition MPC tuning")
    parser.add_argument("--image", default=os.getenv("SDU_APEX_IMAGE", DEFAULT_IMAGE))
    parser.add_argument("--smoke-test", action="store_true", help="Verify controller restart leaves Unity alive and both clean up")
    parser.add_argument("--start", action="store_true", help="Automatically start Unity and the MPC when the tuner opens")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]{8,80}", SESSION):
        parser.error("invalid SDU_APEX_TUNER_SESSION")
    STATE.mkdir(parents=True, exist_ok=True)
    if args.smoke_test:
        smoke_test(args.image)
        return
    root = tk.Tk()
    app = CompetitionTunerGUI(root, args.image)
    if args.start:
        root.after(650, app._start)
    root.mainloop()


if __name__ == "__main__":
    main()
