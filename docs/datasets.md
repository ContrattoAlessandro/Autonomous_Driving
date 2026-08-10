# Public Traffic-Light Datasets — Selection & Justification

This document justifies the dataset corpus for training **YOLO26x-P2** for
real-world traffic-light detection. It is the data chapter of the thesis's
offline evaluation (§6.2).

Each candidate is scored on four axes requested by the task: **size & variety**
(day/night/rain/multi-city), **annotations** (state / pictograms / boxes),
**YOLO compatibility**, and **reported performance**.

---

## 1. Chosen corpus (4 datasets + 1 optional)

### 1.1 DTLD — DriveU Traffic Light Dataset  *(primary)*
- **Source:** Fregin, Müller, Kreßel, Dietmayer — Ulm University / BMW. ICRA 2018.
  DOI [10.1109/ICRA.2018.8460737](https://doi.org/10.1109/ICRA.2018.8460737).
- **Download:** https://www.traffic-light-data.com/  ·  parser: https://github.com/julimueller/dtld_parsing
- **Size:** ~23,000+ frames, **>100,000 annotated lights**.
- **Variety:** **11 German cities** (Berlin, Bochum, Bremen, Dortmund, Düsseldorf,
  Essen, Frankfurt, Fulda, Hannover, Kassel, Köln); variable weather/illumination
  (day / night / rain).
- **Annotations (richest of any public set):** per-light attributes include
  `state` (red / yellow / green / red+yellow / off), **`pictogram`** (arrow
  left/straight/right, pedestrian, tram, bicycle, …), `direction`, `relevance`,
  `occlusion`, `orientation`. YAML label files.
- **YOLO compat:** no first-party YOLO format, but the YAML is clean — conversion
  is a ~50-line script (see `scripts/convert_dtld.py`, scaffolded by
  `dtld_parsing`).
- **Reported perf:** the original paper benchmarks Faster R-CNN; DTLD is now the
  de-facto large-scale state+pictogram benchmark in German TLR research.
- **Role in corpus:** **primary** source of state *and* pictogram supervision;
  sole dataset enabling the Tier-C pictogram experiment.
- **License:** research-use (check the LICENSE file shipped with the download
  before any commercial use).

### 1.2 Bosch Small Traffic Lights Dataset (BSTLD)
- **Source:** Behrendt, Novak, Botros — Heidelberg University (HCI/IWR). ITSC 2017.
  (Hosted under the legacy `bosch-ros-pkg` GitHub org — hence the name.)
- **Download:** https://hci.iwr.uni-heidelberg.de/node/6132  ·  repo https://github.com/bosch-ros-pkg/bstld
- **Size:** ~13,425 test images + training set; ~6,000–7,000 TL boxes (small-object
  focus, many empty frames).
- **Variety:** Germany, urban + rural; **explicitly engineered for far / early
  detection, including self-illuminated lights at night** (day + night).
- **Annotations:** 4 states — **off / green / yellow / red**. Boxes + state.
  **No pictograms.**
- **YOLO compat:** **best of any dataset here** — multiple ready-made converters,
  e.g. https://github.com/DRitchiezx/bosch_traffic_lights_to_yolo .
- **Reported perf:** authors report SSD ~50–60% AP on their day/night split;
  later YOLO/RetinaNet variants reach 70–90% mAP.
- **Role in corpus:** small-object benchmark (perfect fit for the **P2 head**),
  day/night robustness validation, and the only **commercially-permissive** set.
- **License:** **MIT** (commercial-friendly).

### 1.3 LISA Traffic Light Dataset
- **Source:** Jensen, Philipsen, Møgelmose, Moeslund (Aalborg) + Trivedi (UCSD LISA).
  IEEE T-ITS 2016.
- **Download:** http://cvrr.ucsd.edu/LISA/lisa-traffic-light-dataset.html  ·
  Kaggle mirror: https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset
- **Size:** **43,017 frames, 113,888 annotated lights**.
- **Variety:** San Diego, CA, USA; **explicit day + night test split**
  (~24,988 day / ~18,028 night frames).
- **Annotations:** **14 classes** derived from `go / stop / warning` × direction
  (+ a "light vs full housing" toggle). State-rich.
- **YOLO compat:** excellent — 3+ community YOLO repos
  (e.g. https://github.com/AbhishekSinha-git/LISA-traffic-light-dataset-yolov8-Training).
- **Reported perf:** authors' CNN pipeline reaches high-80s / low-90s % state
  recognition accuracy; standard day/night benchmark in TLR literature.
- **Role in corpus:** **US geography complement** (horizontal 3-light stacks vs
  European signals) and a canonical day/night test split.
- **License:** CC BY-NC-SA 4.0 (**non-commercial** — fine for thesis/research).

### 1.4 Open Images V7 — "Traffic light" class
- **Source:** Google. Class list confirmed at
  https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv .
- **Size:** ~14,000–20,000 traffic-light boxes (≈10× COCO's TL count). *Exact count
  to be confirmed via `fiftyone` during conversion (Phase 1 caveat).*
- **Variety:** global crowd-sourced imagery, mostly daytime, mixed signal housings.
- **Annotations:** **bounding box only — NO state, NO pictograms.**
- **YOLO compat:** trivial via `fiftyone` or `oidv6-to-yolo`; extract just the
  "Traffic light" class.
- **Role in corpus:** **large-scale detection pretraining + hard negatives** to
  boost box localization and worldwide housing variety. State-free, so used only
  in Tier A.
- **License:** CC BY 4.0 (commercial-friendly).

### 1.5 LaRa (optional — not in the primary corpus)
- **Source:** Inria Lille / Univ. Lille. IEEE SSCI / IJCNN 2013.
- **Size:** ~11,179 frames from one Paris urban sequence.
- **Variety:** Paris dense urban, daytime only (weak day/night).
- **Annotations:** 3 states (green/red/yellow), no pictograms.
- **Role:** optional extra EU geographic diversity if Tier-B numbers need a
  second European city beyond the DTLD German set.

---

## 2. Datasets considered and rejected

| Dataset | Why rejected |
|---|---|
| **COCO** "traffic light" (#10) | Only ~2–3k TL instances (≈10× fewer than Open Images), **no state**, superseded by OI for detection pretraining. |
| **BDD100K** | TL appears **only as segmentation masks** (id 25), not in the 10-class detection benchmark; masks → bbox loses state info. Useful only as background augmentation. |
| **Waymo Open Dataset** | Richest labels (state + arrow + lane association), but heavy `.tfrecord` extraction friction + non-commercial research license. Not worth the lift for an offline benchmark. |
| **nuScenes** | **No 2D traffic-light bounding boxes** in camera images — TLs are static 3D map polygons only. Useless for YOLO training. |
| **Mapillary Vistas** | Segmentation masks only, no state. Same limitation as BDD. |
| **WPI TL** (2010–2012) | Small, historical; superseded by LISA/DTLD. |

---

## 3. Unified class taxonomy

The three tiers (see `configs/data_tier{A,B,C}.yaml`):

### Tier A — detection only (1 class)
`traffic_light` — all four datasets contribute. Baseline + box-prior pretraining.

### Tier B — state model (6 classes, **primary**)
`red`, `yellow`, `green`, `red_yellow`, `off`, `unknown`

Per-source mapping (`scripts/harmonize_labels.py` encodes this):

| Unified class | LISA native             | Bosch native | DTLD native (`state` attr)            |
|---|---|---|---|
| `red`         | `stop`, `stopLeft`      | `Red`        | `red`                                |
| `yellow`      | `warning`, `warningLeft`| `Yellow`     | `yellow`                             |
| `green`       | `go`, `goLeft`          | `Green`      | `green`                              |
| `red_yellow`  | —                       | —            | `red_yellow`                         |
| `off`         | —                       | `Off`        | `off`                                |
| `unknown`     | anything else           | —            | `unknown` / occluded / irrelevant    |

LISA "light vs full housing" annotations are collapsed to the housing box.

### Tier C — pictogram ablation (DTLD only, ~9 classes)
The 6 Tier-B classes **plus** pictogram subclasses that DTLD labels separately:
`arrow_left`, `arrow_straight`, `arrow_right`, `pedestrian`, … (final list
written by `harmonize_labels.py` after scanning DTLD; rare pictograms merged).

---

## 4. Caveats to verify during Phase 1 (conversion)

The following figures are widely reported but should be re-confirmed against the
downloaded data before being cited in the thesis:
- **DTLD** exact frame / box counts (commonly ~23,758 frames / ~119,000 lights).
- **BSTLD** exact image / box totals (the repo README does not print them; check
  the Heidelberg benchmark page).
- **Open Images V7** exact traffic-light box count (~14–20k).
`scripts/eda.py` writes the verified counts to `results/splits_summary.csv`.
