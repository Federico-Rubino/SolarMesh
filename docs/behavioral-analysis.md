# SolarMesh — How the Behavioral Analysis Works

*Status: 2026-10-10. Describes the software in `software/python/`. The simulator in `simulation/` is only used to test it.*

## 1. What it does

An operator who only sees plant-level production knows *that* output dropped, not *why*. SolarMesh answers two questions continuously, using only data the plant already produces:

1. **Is a human needed?** For every module: is this drop temporary (cloud), a cleaning job (soiling), something to inspect, or a fault that needs a technician?
2. **Is a cloud coming?** From the way shadows move across the modules: speed and direction, and when each module will be shaded.

Everything runs locally on the Arduino UNO Q (Linux side). There's no cloud service and no machine learning: the logic is a set of explainable rules plus a geometric fit.

### Inputs

Every 0.5 s the Linux side asks the STM32 for one snapshot over RPC (`get_mesh`):

| Field | Meaning |
|---|---|
| `nodes` | voltage of each module (V) |
| `current` | reference current from the electronics (A) |
| `temp` | array temperature (°C), displayed only |
| `edges` | `[a, b, wired]`: modules `a` and `b` are physically adjacent; `wired = 1` if they are also electrically connected in series |
| `layout` *(optional)* | position of each module `{id: [x, y]}`, needed for cloud tracking |

No extra sensors are required: no camera, no irradiance meter, no thermal imaging.

## 2. The processing chain

```
get_mesh ─► topology graph ─► module health ─► per-module history ─┬─► human-needed predictor
                                                                    └─► cloud tracker
                                                                              │
                                                              digital twin payload ─► GUI / alerts
```

### 2.1 Topology graph (`topology.py`)

The edges become a NetworkX graph. It serves two purposes:

- **Electrical view** (only `wired` edges): each connected chain is one **string** of modules in series. The software checks that every string is a simple chain and reports loops, branches, unwired modules, or modules that appear in `edges` but were never reported.
- **Spatial view** (all edges): who is next to whom. The diagnosis uses this to tell "one module alone is low" from "a whole area is low".

The plant can have any shape: rectangular, circular or irregular. Only the edges matter.

### 2.2 Module health (`optimizer.py`)

Health is **relative to the best module right now**:

```
health_i = V_i / V_best          (0 … 1)
```

This needs no datasheet or irradiance sensor, and temperature cancels out because all modules share it. Derived figures:

- **Ideal power:** `I × V_best × N`, i.e. every module as good as the best one.
- **Actual power (estimate):** `I × min(health) × ΣV`. Modules in series share one current, so the weakest one limits the string.
- **Power lost by a module:** `I × (V_best − V_i)`, used to rank problems.

### 2.3 Per-module history (`diagnosis.py`)

Each module keeps a short history (10 minutes) and three facts derived from it:

| Fact | How it's computed |
|---|---|
| **Anomalous since** | health drops **below 85%** → start time recorded. It ends only when health rises **above 90%**. The gap prevents flickering around the threshold. |
| **Drop speed** | time between the last healthy sample (**≥ 95%**) and the moment it became anomalous. **≤ 3 s = sudden.** If there is no healthy sample in the history, the decline was **gradual**. |
| **Isolated** | fewer than half of its physical neighbours are anomalous too (a module with no neighbours counts as isolated). |

## 3. Human-needed predictor

### 3.1 Decision rules

Rules are evaluated top to bottom; the first match wins.

| # | Condition | Diagnosis | Action |
|---|---|---|---|
| 1 | not anomalous | `ok` | none |
| 2 | anomalous for **less than 20 s** | `shade` | none |
| 3 | health **≤ 5%** | `dead` | **technician** |
| 4 | **gradual** decline | `soiling` | clean |
| 5 | **sudden** and **isolated** | `fault` | **technician** |
| 6 | **sudden** and neighbours also low | `obstruction` | inspect |

**The underlying idea:** different causes leave different fingerprints in **time** and **space**.

| Cause | Duration | Onset | Spatial pattern |
|---|---|---|---|
| Cloud | short | sudden | a group of neighbours, moving |
| Soiling | persistent | slow, over days | single module or area |
| Fault (crack, connector, diode) | persistent | sudden | one module alone |
| Obstruction (new structure, vegetation) | persistent | sudden or slow | a group of neighbours, static |

### 3.2 Output

- Each module: `diagnosis`, `action`, how long it has been in that state, and its power loss.
- Plant level: `technician.needed` (yes/no) and the list of modules to visit, shown as the red/green **Technician** banner.

### 3.3 Timing

The **20 s** persistence threshold is compressed for the demo, so the effect is visible in front of a jury. On a real plant it should be minutes to hours, so that slow clouds and morning shade never cause alerts. It's a single constant (`PERSIST_S`).

## 4. Leaf vs soiling vs fault

| Case | What the system sees | Current classification |
|---|---|---|
| **Dust / soiling** | slow decline over days, usually several modules at a similar rate, recovers after rain | `soiling` → **clean** ✔ |
| **Fault** | sudden, deep, persistent drop on one module | `fault` → **technician** ✔ |
| **Leaf, bird dropping, plastic bag** | sudden, persistent drop on one module, often only partial | `fault` → **technician** ✘ |

**Known limitation:** with voltage alone, an object that lands on a module looks exactly like a sudden fault: same timing, same isolation. Today the system would send a technician for a leaf. Signals that could separate them, none implemented yet:

- **Depth of the drop:** debris usually covers part of a module (moderate drop); a failed bypass diode or broken connection is usually deeper or a characteristic step.
- **Self-recovery:** leaves and bags often blow away; wind or rain that brings the module back confirms debris.
- **Visual check before dispatch:** in the "sudden + isolated" case, ask for a photo (e.g. via a Telegram bot and on-device vision model) before sending a technician.

A reasonable next rule: *sudden + isolated + moderate drop* → `debris?` → **request visual check**, and send a technician only if the drop is deep or the photo confirms damage.

## 5. Cloud tracking (`radar.py`)

### 5.1 Principle

The modules work as the pixels of a low-resolution camera pointed at the sky. When a cloud shadow crosses the plant, its edge reaches each module at a different time. For a shadow edge moving in a straight line, **the time a module gets shaded grows linearly with its distance along the direction of motion**:

```
t(x, y) = t₀ + sx·x + sy·y
```

`(sx, sy)` is the *slowness* vector (seconds per metre). Fitting this plane to observed shading events gives the cloud's motion.

### 5.2 Steps

1. **Collect onsets:** for every module that became anomalous with a **sudden** drop in the last **30 s**, take its position and the time: `(x, y, t)`. Gradual drops (soiling) are ignored.
2. **Wait for enough evidence:** at least **5** onsets, or as many modules as the plant has if it's smaller (so 2 panels work).
3. **Fit the front** with least squares:
   - if the onsets spread in 2D → full plane fit, giving direction and speed;
   - if they lie on one line (e.g. a 2-panel bench) → fit along that line only, giving speed along that axis.
4. **Quality checks**, rejecting the fit when:
   - the slowness is ~0, i.e. everything was shaded at the same instant (no measurable motion);
   - the points don't fit a moving front (RMS error **> 1 s**), e.g. confused events from two clouds or a whole-plant overcast.
5. **Motion:** velocity `v = s / |s|²`; speed in m/s (times module pitch); heading in degrees (x to the right, y down in plant coordinates).
6. **Predict arrival:** for every module not yet shaded, `ETA = t(x, y) − now`, reported if it's between 0 and **60 s**.

### 5.3 Output

- `cloud.active`, `speed_mps`, `heading_deg`, `eta` per module, `next_eta`.
- GUI: **"Cloud approaching: first module in N s"** banner, a dashed ring with a countdown on each module about to be shaded, and an arrow showing the direction of motion.

### 5.4 Warning time depends on plant size

The tracker only sees a shadow once it reaches the plant, so the warning time is about **plant size ÷ cloud speed**: a few seconds on a 25 m rooftop, minutes on a kilometre-scale utility plant. Sharing data between neighbouring plants would extend it further.

## 6. What the operator sees

| Element | Content |
|---|---|
| **Technician banner** | red: modules to visit + watts lost · amber: cleaning or inspection only · green: no technician needed |
| **Cloud banner** | approaching (ETA, modules affected, speed, heading) · over the plant · sky clear |
| **Digital twin** | each module coloured by diagnosis (or health/voltage), icons per cause, cloud countdowns and arrow, electrical strings drawn as lines |
| **Module detail** | diagnosis, recommended action, duration, voltage, health, power lost, cloud ETA, string and position, neighbours, voltage history |
| **Alerts** | readable messages, e.g. `n248: fault for 22 s → send technician` |

## 7. Measured performance

Tested by feeding the unmodified software a simulated 500-module plant (25 × 20 m, 20 strings) with 12 clouds from all directions at 2–12 m/s, plus one injected module fault per run:

| Metric | Result |
|---|---|
| Cloud speed error | 1.2% |
| Cloud heading error | 7.9° mean |
| Modules warned before being shaded | 96% (median 2.9 s ahead) |
| Arrival-time error (warning < 5 s ahead) | 0.24 s |
| Arrival-time error (warning 5–15 s ahead) | 1.8 s |
| Fault → technician alert | after 19.8 s (the 20 s persistence rule) |
| False technician alerts caused by clouds | **0** |
| Compute time for 500 modules | ~0.3 s per update on the UNO Q |

These figures come from a simulation that assumes module voltage drops in proportion to light. They still need confirming on real panels (see §8).

## 8. Limitations and next steps

1. **What voltage is measured.** A panel's open-circuit voltage barely changes in shade; under load it drops a lot. The detection thresholds assume the STM32 reports a voltage that responds strongly to shade. *Needs confirmation from the electronics team.*
2. **Relative health.** Because the reference is the best module *now*, a cloud covering the whole plant makes everything look healthy, and the shading events when it leaves are confused. This causes the remaining heading errors (up to ~20°) and could later make a long overcast look like an obstruction. *Fix: also compare against the recent peak voltage, and treat "most of the plant low at once" as weather (no action).*
3. **Leaf vs fault.** See §4: currently a leaf triggers a technician alert.
4. **Positions.** Cloud tracking needs real module positions (`layout`). Without them the twin draws each string as a row, which is fine for display but not for motion estimates.
5. **Sampling rate.** At 2 readings per second, very fast shadows on small plants are hard to time. Faster sampling on the STM32 (≥ 10–20 Hz) would sharpen both the tracker and the "sudden vs gradual" decision.
6. **Thresholds.** 85% / 90% / 95%, 3 s, 20 s are demo values, chosen by hand. They should be tuned on real data, for example with labelled scenarios from the simulator.
