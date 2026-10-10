# SolarMesh — Realistic Farm Model (testbench)

*Status: 2026-10-10. Code: `simulation/farm.py` (model) and `simulation/testbench.py` (benchmark). Simulation only; the real software in `software/` never imports it.*

## Purpose

The benchmark must not flatter the system. The farm model therefore reproduces how shading **really** shows up in per-module voltage: through bypass diodes and the inverter's maximum-power tracking. It doesn't just make voltage proportional to light.

## Plant

| Parameter | Value | Basis |
|---|---|---|
| Capacity | ~1 MWp | target |
| Module | 550 W, 2278 × 1134 mm, Voc 49.7 V, Vmp 42.0 V, Isc 13.9 A | 550 W datasheets (DAS DH144PA, Canadian CS6W, SEG 550) |
| Temperature coefficient (voltage) | −0.26 %/°C | Canadian CS6W datasheet |
| Bypass diodes | 3 per module (3 substrings of 24 cells) | standard 144 half-cell layout |
| String length | 26 modules | Voc at low temperature vs. the 1500 V limit |
| Layout | 35 rows × 52 modules (2 strings per row), modules side by side, 1.154 m apart | — |
| Row pitch | 5.06 m (2.278 m collector at ground coverage ratio 0.45) | LBNL: fixed-tilt GCR ~0.40–0.50 |
| Total | 1,820 modules, 70 strings, 59 × 172 m | 1 MWp / 550 W ≈ 1,820 |

## Electrical model

- **Each substring** uses a single-diode model, `V = n·Ns·Vt·ln((Iph − I)/I0 + 1) − I·Rs`, with I0 and Rs fitted to the datasheet Voc (49.7 V) and Vmp (42.0 V).
- **Bypass:** if the string current is above what a substring can produce, its bypass diode conducts and the substring contributes **−0.7 V**. One shaded substring drops a module by about **1/3** (NREL, Deline); a fully shaded module reads about −2 V.
- **Inverter MPPT** per string: the current that maximises string power is chosen among the candidate light levels. With a few shaded modules the inverter keeps full current and bypasses them. When most of the string is shaded it lowers the current instead, and all modules then read similar, normal-looking voltages.
- **Cell temperature:** ambient 25 °C + (NOCT 45 − 20)/800 × irradiance.
- **Measurement noise:** σ = 0.05 V per module reading.
- **What the STM32 reports:** each module's voltage under load, and the mean string current.

## Weather

| Parameter | Value | Basis |
|---|---|---|
| Clear-sky irradiance | 900 W/m² | midday, assumption |
| Cloud shadow speed | 5–15 m/s per scenario (±15% per cloud), heading ±10° around the wind | shadow-speed sensor study (Fung 2014); altocumulus case 14.3 m/s |
| Shadow size | radius 100–600 m | cumulus shadows are power-law distributed, hundreds of metres to km |
| Light under the shadow | 25–50% of clear sky | direct beam blocked, diffuse remains (assumption, not directly confirmed) |
| Shadow edge | 50 m transition, with up to +10% brightening at the edge | dense pyranometer network: sharp 40–60 m transitions, edge enhancement up to +20% |
| Overcast | ramp to 30% of clear sky over 2 minutes | clearness index of cloudy skies 0.1–0.3 |

## Events (ground truth)

| Event | Model | Correct action |
|---|---|---|
| Bypass-diode short / cracked substring | one substring permanently at 0 V (module ≈ −1/3) | technician |
| Dead module | no photocurrent, all diodes conduct (≈ −2 V) | technician |
| Debris (leaf, dropping) | one substring at 30% light | clean |
| Tree/structure shadow | static 2 × 3 module patch at 30% light | inspect |
| Local soiling | two edge rows lose up to 30% over the run (**time-compressed**: real rates are 0.05–1 %/day) | clean |
| Cloud shade / overcast | weather | none |

Field failure rate for context: about 5 failures per 10,000 modules per year (NREL/DuraMat), i.e. about one fault per year on this 1 MWp plant. The testbench injects faults far more often, to measure detection.

## Scenarios

`cumulus_0 … cumulus_300` (6 wind directions, each with faults injected at 60, 90 and 120 s), `overcast`, `tree_shadow`, `local_soiling`, `clear_sky`. Each runs for 600 s at 0.5 s per sample, under two threshold settings:

- **demo:** a problem must persist 20 s;
- **field:** it must persist 300 s.

## Known model limits

- No row-to-row shading, no albedo/bifacial gain, no inverter clipping.
- The light level under a cloud and the edge width are plausible, not measured on site.
- Soiling is time-compressed, so the "gradual vs sudden" timing differs from reality.
- The same model generates the data and defines the truth, so results show internal consistency, not field validation.

## Sources

- [DAS-DH144PA 550 W datasheet](https://sun.store/en/product/das-solar-das-dh144pa-550-540-560-silver-frame-35276) · [Canadian CS6W-550MB-AG](https://sun.store/en/product/canadian-cs6w-550mb-ag-silver-frame-1500v-3019) · [SEG-550-BMA-TB](https://portlandiaelectric.supply/product/seg-550w-bifacial-silver-frame-144-mono-cells-solar-module-seg-550-bma-tb)
- [LBNL – Land requirements for utility-scale PV (GCR)](https://emp.lbl.gov/sites/default/files/emp-files/land_requirements_for_utility-scale_pv.pdf) · [pv magazine – inter-row spacing guidelines](https://www.pv-magazine.com/2023/07/06/new-guidelines-for-inter-row-spacing-of-pv-power-plants/) · [SurgePV – land per MWp](https://www.surgepv.com/blog/solar-land-area-calculator)
- [Deline (NREL) – partially shaded grid-tied PV, 1/3 voltage drop per bypass diode](https://kirschcenter.deanza.edu/faculty/hamidiridha/esci61/documents/chris-deline-partially-shaded-operation-of-a-grid-tied-pv-system.pdf)
- [Observed patterns of surface irradiance under cloudy skies (40–60 m transitions)](https://arxiv.org/pdf/2307.06980) · [Cloud size distributions and irradiance variability](https://arxiv.org/pdf/2209.10284) · [Witthuhn et al., ACP 2026](https://acp.copernicus.org/articles/26/5727/2026/) · [Fung et al. 2014, cloud shadow speed sensor](https://amt.copernicus.org/articles/7/1693/2014/)
- [Clearness index ranges](https://greenrhinoenergy.com/solar/radiation/atmosphere.php)
- [PV Tech – soiling in dry climates (0.1–1 %/day)](https://pv-tech.org/pv-soiling-in-dry-climates-causes-impacts-and-solutions) · [Spanish soiling field study](https://ideas.repec.org/a/eee/renene/v156y2020icp743-747.html)
- [DuraMat/NREL – ~5 failures per 10,000 modules per year](https://www.duramat.org/docs/duramatlibraries/assets/working-with-duramat/2017-05-09-jordan.pdf) · [NREL degradation review (median 0.5 %/yr)](https://research-hub.nrel.gov/en/publications/photovoltaic-degradation-rates-an-analytical-review-2/)
