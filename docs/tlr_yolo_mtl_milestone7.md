# TLR-YOLO-MTL — Milestone 7: training multi-dataset

**Stato corrente:** trainer implementato; training finale DTLD paired non
ancora eseguito. La metodologia canonica è in
`metodologia_pipeline_attuale.md`.

## Pipeline attiva

È disponibile un trainer dedicato al modello completo P3–P5. La configurazione
attiva di prototipazione usa YOLO11n; YOLO11l resta disponibile solo per una
successiva fase sperimentale. Il trainer non modifica lo script Ultralytics
preesistente `scripts/train.py` e usa direttamente il manifest canonico.

Impostazioni resource-aware predefinite:

- input fisso 800×1600;
- micro-batch 16 validato sulla RTX 5070;
- batch effettivo 32 con 2 passi di accumulo;
- 100 optimizer step bilanciati per epoca, cioè 3.200 immagini campionate;
- `workers=2` persistenti e `prefetch_factor=1` su Windows;
- AMP FP16 con scala iniziale 32, gradient clipping 10 ed EMA 0,9999;
- AdamW con LR distinto per backbone e neck/teste;
- nessuna P2.

Le quote su ogni finestra di accumulo sono state adattate ai dataset realmente
disponibili:

| Gruppo | Immagini / 32 |
|---|---:|
| DTLD paired | 26 |
| ATLAS + LISA | 6 |

Bosch, CeyMo e TLD-READY non compaiono nel sampler.

## Loss unificata

Il criterio calcola una sola assegnazione YOLO per i semafori e combina:

- DFL + CIoU della detection;
- NWD ausiliaria con `C=12` e peso 0,5;
- focal cross-entropy per stato e pittogramma, `gamma=1,5`;
- detection e focal BCE multi-label delle frecce;
- focal BCE della relevance, `gamma=2`;
- task mask per immagine e per istanza.

Le candidate background il cui centro cade in una `ignore_region` vengono
rimosse dalla BCE di classificazione; non sono trasformate in target positivi.
Le 28.525 immagini DTLD official-train hanno annotazioni freccia esaustive: i
14.855 file vuoti contribuiscono correttamente come negativi. ATLAS, LISA e il
test ufficiale DTLD restano mascherati per la loss frecce.

## Quattro fasi

La configurazione `configs/tlr_yolo_mtl_train.yaml` definisce:

1. 5 epoche di warm-up con backbone e neck congelati, contesto disabilitato;
2. 25 epoche di perception pretraining;
3. 80 epoche joint con contesto denso abilitato;
4. 20 epoche di fine-tuning con LR ridotto e statistiche BatchNorm congelate.

Con 100 step per epoca, le quattro fasi corrispondono a 13.000 optimizer step e
416.000 presentazioni di immagini. Un passaggio nominale sui 70.285 record di
train corrisponderebbe a circa 2.197 finestre effettive da 32; il sampler usa
comunque quote e replacement, quindi l'epoca è definita dal budget di step e
non da una visita esatta di ogni record. Il nano rende rapido il ciclo di
prototipazione senza cambiare le quote supervisionate o l'architettura
multi-task; il modello large potrà essere riattivato dopo la validazione.

Il trainer salva `last.pt` atomicamente e checkpoint periodici ogni 10 epoche.
I checkpoint a fine epoca includono modello, EMA, optimizer, scheduler, scaler e
stato RNG e possono essere ripresi con `--resume`. Un checkpoint prodotto da
una prova interrotta a metà epoca viene marcato come non riprendibile, evitando
di saltare in silenzio parte del sampler.
La selezione scientifica di `best.pt` richiederà le metriche validation del
Milestone 8, non la training loss.

## Stabilità numerica del warm-up

Il primo avvio nano ha mostrato una loss `NaN` dopo cinque optimizer step. La
causa era nella policy di freeze: `warmup_heads` congelava il backbone ma stava
aggiornando anche il neck con il learning rate alto delle teste. Ora il warm-up
allena soltanto la testa finale; nelle fasi successive tutto il feature extractor
(backbone + neck) usa `backbone_lr`.

Il trainer interrompe inoltre immediatamente il backward se una componente
della loss forward non è finita e salva `failure.json`, evitando cicli infiniti
di dimezzamento della scala AMP. La correzione è stata verificata per un'epoca
completa: 100 optimizer step, loss media da 301,10 a 109,05, scala AMP stabile a
32 e checkpoint riprendibile. Anche la ripresa allo step 101 è stata verificata.
Report: `results/tlr_yolo_mtl/prototype_nano_stability_epoch.json`.

## Relevance e frecce paired

Il nuovo corpus contiene 28.525 immagini DTLD con relevance e annotazioni
freccia esaustive sulla stessa immagine. Su questi record il collegamento
relevance→arrow usa la scala controllata 0,25; su ATLAS, LISA e sul test
ufficiale DTLD resta a zero. La testa frecce impara direttamente da DTLD, anche
dalle immagini negative verificate. Il contratto completo è descritto in
`docs/tlr_yolo_mtl_milestone10_dtld_paired.md`.

## Smoke reale

I report di smoke attualmente salvati su DTLD, ATLAS e CeyMo documentano il
vecchio corpus unpaired e sono conservati solo come verifica storica. Lo script
corrente seleziona invece una DTLD paired e una ATLAS/LISA e richiede scala
contesto 0,25 per l'immagine DTLD. Questo smoke GPU deve essere rieseguito e il
report sovrascritto prima del training paired finale.

Sul vecchio corpus era stato inoltre eseguito un probe AMP reale nella fase più
pesante con batch 16
e shape `[16, 3, 800, 1600]`, includendo backward, clipping, uno step AdamW e un
aggiornamento EMA. Il picco è 8.859.077.120 byte allocati e 9.743.368.192 byte
riservati su 12.820.480.000 byte disponibili. Lo step è finito e applicato con
scala AMP 32. Restano circa 3,08 GB rispetto alla memoria totale.

Report: `results/tlr_yolo_mtl/prototype_nano_memory_probe_batch16.json`.

Il loop completo è stato verificato con tre optimizer step. Con `workers=2` e
`prefetch_factor=1` il trial ha richiesto 51,4 secondi totali, inclusi avvio dei
worker e checkpoint; con `workers=0` ha richiesto 54,9 secondi. I checkpoint
temporanei da 39,3 MB,
non riprendibili perché fermati a metà epoca, sono stati rimossi dopo aver
salvato il report compatto in
`results/tlr_yolo_mtl/prototype_nano_trainer_trials.json`.

## Comandi

Smoke limitato:

```powershell
.\.venv\Scripts\python.exe -B -m scripts.check_tlr_yolo_mtl_training
```

Probe memoria alla risoluzione finale:

```powershell
.\.venv\Scripts\python.exe -B -m scripts.check_tlr_yolo_mtl_training_memory `
  --batch 16 --output results\tlr_yolo_mtl\prototype_nano_memory_probe_batch16.json
```

Prova reale di un optimizer step (checkpoint volutamente non riprendibile):

```powershell
.\.venv\Scripts\python.exe -B -m scripts.train_tlr_yolo_mtl `
  --config configs\tlr_yolo_mtl_train.yaml --batch 4 `
  --phase warmup_heads --max-optimizer-steps 1 `
  --output-dir runs\tlr_yolo_mtl_nano_dtld_paired_trial
```

Training completo:

```powershell
.\.venv\Scripts\python.exe -B -m scripts.train_tlr_yolo_mtl `
  --config configs\tlr_yolo_mtl_train.yaml --batch 4 --steps-per-epoch 100 `
  --output-dir runs\tlr_yolo_mtl_nano_dtld_paired_seed42
```

Ripresa del nuovo esperimento paired, dall'ultima epoca completa:

```powershell
.\.venv\Scripts\python.exe -B -m scripts.train_tlr_yolo_mtl `
  --resume runs\tlr_yolo_mtl_nano_dtld_paired_seed42\weights\last.pt
```

Il checkpoint del vecchio run `tlr_yolo_mtl_nano_seed42_v2` non è compatibile
con il nuovo contratto dati/quote e non deve essere usato con `--resume`.

## Stato

Implementazione e backward multi-dataset completati. Il Milestone 7 sperimentale
non è ancora concluso: non sono state lanciate le 130 epoche né prodotti i tre
seed finali, coerentemente con il vincolo di risorse e con la necessità di
completare prima la validazione automatica.
