# TLR-YOLO-MTL — Milestone 10: DTLD paired definitivo

**Stato:** milestone storica del corpus. La decisione paired resta valida, ma
la quota `26 DTLD + 6 ATLAS/LISA`, il doppio detector e FiLM sono stati
sostituiti dalla mainline DTLD-only con detector unificato e cross-attention.

**Riferimento metodologico:** `metodologia_pipeline_attuale.md`.

## Decisione

`dataset_ALL_USER_ANNOTATED` sostituisce sia TLD-READY sia CeyMo nella pipeline
multi-task. Le annotazioni umane delle frecce vengono fuse nei record DTLD
originali, quindi ogni immagine paired mantiene contemporaneamente:

- box, stato e pittogramma del semaforo;
- relevance del semaforo;
- box e direzione multi-label delle frecce stradali.

ATLAS e LISA restano sorgenti ausiliarie per la percezione dei semafori. Bosch
resta escluso perché il download disponibile non era completo. I dati CeyMo
eventualmente presenti sul disco non vengono cancellati, ma non sono importati,
campionati o accettati dal contratto di training.

## Copertura e tassonomia

La directory annotata contiene 28.525 immagini e 28.525 file label. La
copertura coincide esattamente con tutte le immagini dello split ufficiale
DTLD `train`:

| Voce | Quantità |
|---|---:|
| immagini annotate | 28.525 |
| immagini positive | 13.670 |
| immagini negative verificate | 14.855 |
| box freccia | 31.528 |
| straight | 15.805 |
| left | 7.740 |
| right | 2.874 |
| straight-left | 998 |
| straight-right | 4.111 |

I file label vuoti sono negativi esaustivi, non annotazioni mancanti. La
direzione è codificata `[left, straight, right]`; le classi composte diventano
rispettivamente `[1,1,0]` e `[0,1,1]`.

Il converter richiede corrispondenza uno-a-uno tra immagini, label e record
DTLD, verifica le dimensioni, rifiuta classi/righe/coordinate invalide e non
consente sovrascritture silenziose. Quattordici box oltrepassavano realmente il
bordo e sono stati clippati conservando la correzione nei metadati. Altri 401
clamp inferiori a 0,01 pixel derivano soltanto dall'arrotondamento a sei cifre
del formato YOLO e sono tracciati separatamente.

I JPEG puliti richiesti dal converter canonico si rigenerano dai TIFF originali
con `python -m scripts.prepare_dtld_images`. Il parser `dtld_parsing` è fissato
a commit in `requirements.txt`; lo script non offre una modalità con box
disegnati e rifiuta sempre una destinazione non vuota. Il comando completo è
riportato in `COMMANDS.md`.

## Corpus generato

Il corpus canonico si trova in `datasets/tlr_mtl_dtld_paired`:

| Sorgente | Train | Validation | Test | Totale |
|---|---:|---:|---:|---:|
| DTLD | 22.563 | 5.962 | 12.453 | 40.978 |
| ATLAS | 27.187 | 3.029 | 2.828 | 33.044 |
| LISA | 20.535 | 0 | 22.481 | 43.016 |
| Totale | 70.285 | 8.991 | 37.762 | 117.038 |

Tutti i 22.563 DTLD train e i 5.962 DTLD validation sono paired. I 12.453
record del test ufficiale DTLD non erano presenti nel progetto di annotazione:
rimangono quindi validi per semafori/relevance ma con `arrow_detection=false`.
Non vengono trasformati artificialmente in negativi.

Il QA finale riporta zero record CeyMo, zero anomalie normalizzate e zero
leakage di ID, sequenza o percorso tra gli split.

## Effetto sul training

La finestra effettiva da 32 usa ora:

| Gruppo | Immagini |
|---|---:|
| DTLD paired | 26 |
| ATLAS + LISA | 6 |

Sui record DTLD paired la scala del gradiente relevance→arrow è 0,25. Sui
record non paired resta zero. La testa frecce riceve inoltre la propria loss
supervisionata su tutte le immagini DTLD train, inclusi i negativi esaustivi.

Il precedente checkpoint `tlr_yolo_mtl_nano_seed42_v2` appartiene al contratto
unpaired con CeyMo e non deve essere ripreso con `--resume`. Il training paired
deve iniziare come nuovo esperimento:

```powershell
.\.venv\Scripts\python.exe -B -m scripts.train_tlr_yolo_mtl `
  --config configs\tlr_yolo_mtl_train.yaml `
  --batch 4 `
  --output-dir runs\tlr_yolo_mtl_nano_dtld_paired_seed42
```

Con `--batch 4`, il trainer imposta otto passi di accumulo e conserva il batch
effettivo 32. Il modello resta YOLO11n P3–P5 senza testa P2.

## Limite residuo

È ora possibile addestrare e selezionare il modello usando dati paired reali.
Resta però assente un **test ufficiale DTLD paired**: per una misura finale
indipendente dell'effetto causale delle frecce sulla relevance sarà necessario
annotare anche il test ufficiale oppure congelare, prima degli esperimenti
finali, un holdout paired per sequenza ricavato dai 28.525 frame verificati.
