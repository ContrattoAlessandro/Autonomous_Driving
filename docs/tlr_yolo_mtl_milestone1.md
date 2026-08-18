# TLR-YOLO-MTL — Milestone 1: contratto dati unificato

**Stato corrente:** completato sul corpus DTLD paired.

**Riferimento canonico:** `metodologia_pipeline_attuale.md`.

## Corpus definitivo

Il contratto dati loss-aware usa tre sorgenti:

| Sorgente | Immagini | Supervisione usata |
|---|---:|---|
| DTLD | 40.978 | semaforo, stato, pittogramma, relevance, ignore region; frecce su official-train |
| ATLAS | 33.044 | semaforo, stato, pittogramma |
| LISA | 43.016 | semaforo, stato, pittogramma, giorno/notte |
| **Totale** | **117.038** | — |

Le annotazioni umane in `dataset_ALL_USER_ANNOTATED` sono fuse direttamente
nei 28.525 record DTLD official-train, creando dati paired relevance-frecce.
CeyMo, TLD-READY e Bosch non fanno parte del manifest o del sampler.

## Contratto

Lo schema unificato v2.1 (`ImageRecord`) conserva:

- path e dimensioni dell'immagine;
- split e gruppo di sequenza;
- box dei semafori e delle frecce;
- stato, pittogramma, relevance, occlusione e direzioni multi-label;
- task mask per immagine e validity mask per istanza;
- ignore region e metadati di conversione.

Le immagini originali vengono referenziate e non duplicate. Per DTLD è
obbligatoria una directory di JPEG puliti: il converter rifiuta immagini con
annotazioni disegnate nei pixel.

## Split e QA

Gli split correnti sono 70.285 train, 8.991 validation e 37.762 test. I test
ufficiali sono preservati; DTLD e ATLAS sono divisi per gruppi di sequenza o
blocchi temporali. Il QA corrente riporta:

- 369.522 semafori;
- 31.528 frecce stradali;
- 104.812 ignore region;
- zero duplicati di annotazione;
- zero leakage di ID, sequenza o percorso;
- zero immagini mancanti.

La copertura frecce DTLD comprende 13.670 immagini positive e 14.855 negativi
esaustivi. DTLD official-test non è paired e mantiene la loss frecce mascherata.

## Riproduzione

```powershell
.\.venv\Scripts\python.exe -B -m tlr_yolo_mtl prepare `
  --output datasets\tlr_mtl_dtld_paired --skip-overlays
```

Output:

```text
datasets/tlr_mtl_dtld_paired/
├── manifest.json
├── records.jsonl
├── splits.json
└── qa_report.json
```

## Nota storica

La prima versione della milestone conteneva DTLD, ATLAS, CeyMo e LISA per
119.925 immagini. È stata superata dal paired DTLD del Milestone 10: i numeri e
i report del vecchio corpus non devono essere usati come descrizione della
pipeline attuale.
