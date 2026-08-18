# Dataset della pipeline TLR-YOLO-MTL

**Stato:** documento corrente, allineato al corpus DTLD paired.

**Configurazione canonica:** `configs/tlr_yolo_mtl_data.yaml`.

## Corpus attivo

La pipeline multi-task usa esclusivamente DTLD, ATLAS e LISA. Le immagini sono
referenziate dal manifest e non duplicate.

| Sorgente | Train | Validation | Test | Totale | Task validi |
|---|---:|---:|---:|---:|---|
| DTLD | 22.563 | 5.962 | 12.453 | 40.978 | detection, stato, pittogramma, relevance; frecce su train/val |
| ATLAS | 27.187 | 3.029 | 2.828 | 33.044 | detection, stato, pittogramma |
| LISA | 20.535 | 0 | 22.481 | 43.016 | detection, stato, pittogramma |
| **Totale** | **70.285** | **8.991** | **37.762** | **117.038** | — |

Il QA corrente conta 369.522 semafori, 31.528 frecce stradali e 104.812
ignore region.

## DTLD paired: sorgente principale

DTLD fornisce box, stato, pittogramma, relevance, occlusione e orientamento.
Le immagini pulite in `../DTLD_jpg_plain` sono rigenerate dai TIFF originali;
la pipeline rifiuta immagini preview con box o testo impressi nei pixel.

Le annotazioni umane in `../dataset_ALL_USER_ANNOTATED` coprono esattamente le
28.525 immagini dell'official-train DTLD:

- 13.670 immagini con almeno una freccia;
- 14.855 immagini negative esaustive;
- 31.528 box;
- `straight`: 15.805;
- `left`: 7.740;
- `right`: 2.874;
- `straight-left`: 998;
- `straight-right`: 4.111.

Il converter richiede corrispondenza uno-a-uno tra immagini, label utente e
record DTLD. I file label vuoti sono negativi verificati, non annotazioni
mancanti. Le frecce sono fuse nei record DTLD, producendo supervisione paired
per relevance e frecce sulla stessa immagine.

I 12.453 record DTLD official-test non sono coperti dalle annotazioni frecce:
restano validi per semafori e relevance con `arrow_detection=false`.

## ATLAS: sorgente ausiliaria

ATLAS aggiunge diversità di camera, distanza e configurazione dei semafori. La
sua ontologia nativa a 25 classi viene fattorizzata nelle tassonomie comuni di
stato e pittogramma. Le dimensioni sono sempre lette dal JPEG reale perché
18.639 immagini non usano il canvas nominale 1920x1200.

Il test nativo è preservato. La validation viene ricavata dal train per blocchi
temporali sincronizzati tra le camere, evitando frame quasi duplicati tra
split.

## LISA: sorgente ausiliaria e dominio giorno/notte

LISA aggiunge il dominio statunitense e sequenze giorno/notte. La pipeline usa
le sequenze `dayTrain` e `nightTrain` per il train e conserva le sequenze
ufficiali day/night per il test. Non viene ricavata una validation LISA.

Le annotazioni duplicate della stessa box sono fuse quando compatibili; i
conflitti di pittogramma vengono mascherati invece di forzare un target.

## Tassonomie e mascheramento

| Task | Tassonomia |
|---|---|
| stato | `red`, `yellow`, `green`, `off` |
| pittogramma | `round`, `left`, `straight`, `right` |
| frecce | multi-hot `[left, straight, right]` |
| relevance | binaria per semaforo |

`red_yellow`, stati sconosciuti, pittogrammi composti non rappresentabili e
label mancanti producono target `-1` e non contribuiscono alla loss relativa.
Semafori non veicolari e back-facing diventano ignore region.

## Politica degli split

- preservare sempre i test ufficiali;
- separare DTLD per città, route e sequenza;
- stratificare la validation DTLD per direzione delle frecce;
- separare ATLAS per blocco temporale sincronizzato;
- impedire leakage di ID, sequenza e percorso;
- eseguire QA di geometria, classi, task mask e immagini mancanti.

Il report generato è `datasets/tlr_mtl_dtld_paired/qa_report.json`.

## Sorgenti escluse dalla mainline

| Sorgente | Motivo |
|---|---|
| CeyMo | sostituito dalle annotazioni paired e verificate su DTLD |
| TLD-READY | annotazioni non ottenibili; sostituito dal paired DTLD |
| Bosch Small Traffic Lights | download disponibile incompleto e rimosso per decisione progettuale |
| Open Images | detection-only; non necessario nel contratto multi-task corrente |
| COCO | non è un dataset della pipeline; i suoi pesi sono usati solo per warm-start |

I dati eventualmente presenti sul disco non entrano nel manifest, nel sampler
o nelle loss della pipeline corrente.

## Riproduzione

Da `tl_detection/`:

```powershell
# Solo per rigenerare JPEG puliti in una nuova directory vuota
.\.venv\Scripts\python.exe -B -m scripts.prepare_dtld_images `
  --data-path ..\DTLD `
  --label-path ..\DTLD\v2.0 `
  --target-path ..\DTLD_jpg_plain_new

# Rigenera manifest, split e QA
.\.venv\Scripts\python.exe -B -m tlr_yolo_mtl prepare `
  --output datasets\tlr_mtl_dtld_paired --skip-overlays
```

Per il razionale completo vedere `docs/metodologia_pipeline_attuale.md` e
`docs/tlr_yolo_mtl_milestone10_dtld_paired.md`.
