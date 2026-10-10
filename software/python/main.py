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


def report(payload):
    m = payload["metrics"]
    tech = payload["technician"]
    cloud = payload["cloud"]
    print(
        f"P={m['power_actual']:.2f}/{m['power_ideal']:.2f} "
        f"eff={m['efficiency']:.0%} "
        f"technician={tech['modules'] if tech['needed'] else 'no'} "
        f"cloud={'in %.1fs' % cloud['next_eta'] if cloud.get('next_eta') is not None else 'no'}",
        flush=True,
    )


def step():
    try:
        payload = twin.update(state, link.get_mesh(), time.monotonic())
        if ui:
            ui.send_message("twin", payload)
        report(payload)
    except (OSError, RuntimeError, ValueError, KeyError) as e:
        print(f"error: {e}", flush=True)
    time.sleep(PERIOD)


if __name__ == "__main__":
    if App:
        App.run(user_loop=step)
    else:
        while True:
            step()
