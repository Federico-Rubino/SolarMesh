import time

import link
import twin

try:
    from arduino.app_utils import App
    from arduino.app_bricks.web_ui import WebUI
except ImportError:
    App = None
    WebUI = None

PERIOD = 0.5

ui = WebUI() if WebUI else None
state = twin.new_twin()
latest = {"cleaner": None}


def report(payload):
    m = payload["metrics"]
    tech = payload["technician"]
    cloud = payload["cloud"]
    print(
        f"P={m['power_actual']:.2f}/{m['power_ideal']:.2f} "
        f"eff={m['efficiency']:.0%} "
        f"technician={tech['modules'] if tech['needed'] else 'no'} "
        f"ai_technician={payload['ai']['technician'] if payload['ai']['enabled'] else 'off'} "
        f"cloud={'in %.1fs' % cloud['next_eta'] if cloud.get('next_eta') is not None else 'no'}",
        flush=True,
    )


def as_dict(message):
    if isinstance(message, (list, tuple)) and len(message) == 1:
        message = message[0]
    return message if isinstance(message, dict) else {}


def on_clean(sid, message):
    msg = as_dict(message)
    cleaner = latest["cleaner"] or {}
    try:
        if msg.get("panel") != cleaner.get("panel"):
            raise ValueError(f"panel {msg.get('panel')} has no cleaning motor")
        ms = int(float(msg["seconds"]) * 1000)
        ok = link.clean(ms)
        print(f"clean {msg['panel']}: motor on for {ms} ms: {'ok' if ok else 'rejected'}", flush=True)
        ui.send_message("clean_status", {"ok": bool(ok), "panel": msg["panel"], "ms": ms})
    except (OSError, RuntimeError, ValueError, KeyError) as e:
        ui.send_message("clean_status", {"ok": False, "error": str(e)})


def on_stop_clean(sid, message):
    try:
        link.stop_clean()
        ui.send_message("clean_status", {"ok": True, "stopped": True})
    except (OSError, RuntimeError) as e:
        ui.send_message("clean_status", {"ok": False, "error": str(e)})


if ui:
    ui.on_message("clean", on_clean)
    ui.on_message("stop_clean", on_stop_clean)


def step():
    started = time.monotonic()
    try:
        mesh = link.get_mesh()
        now = mesh["time"] if mesh["time"] is not None else time.monotonic()
        payload = twin.update(state, mesh, now)
        latest["cleaner"] = payload.get("cleaner")
        if ui:
            ui.send_message("twin", payload)
        report(payload)
    except (OSError, RuntimeError, ValueError, KeyError) as e:
        print(f"error: {e}", flush=True)
    time.sleep(max(0.0, PERIOD - (time.monotonic() - started)))


if __name__ == "__main__":
    if App:
        App.run(user_loop=step)
    else:
        while True:
            step()
