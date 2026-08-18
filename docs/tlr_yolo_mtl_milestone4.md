# TLR-YOLO-MTL — Milestone 4: frecce stradali

**Stato corrente:** implementato sul corpus DTLD paired; CeyMo è escluso.

## Architettura

In parallelo alla testa semafori è presente una testa YOLO P3-P5 indipendente:

- detection binaria `road_arrow`;
- tre logits sigmoid per `left`, `straight`, `right`;
- supporto multi-label per `straight-left` e `straight-right`;
- nessuna testa P2.

Gli indici conservati dopo candidate selection e NMS raccolgono i logits di
direzione dalla stessa posizione della detection freccia.

## Supervisione DTLD paired

Le 28.525 immagini DTLD official-train sono annotate in modo esaustivo:

- 13.670 positive;
- 14.855 negative verificate;
- 31.528 frecce.

La loss detection è quindi valida anche sulle immagini senza frecce. La
direzione usa soltanto i match positivi del `TaskAlignedAssigner` e target
multi-hot `[left, straight, right]`.

ATLAS, LISA e DTLD official-test hanno `arrow_detection_valid=false`: non sono
trasformati in negativi e non producono gradiente sulla testa frecce.

## Verifica

```powershell
.\.venv\Scripts\python.exe -B -m scripts.check_tlr_yolo_mtl_arrows
```

Il controllo verifica forward P3-P5, assegnazione, target composti e gradiente
nullo quando il task immagine è mascherato. Report:
`results/tlr_yolo_mtl/milestone4_arrows.json`.

Il report attualmente salvato è una verifica storica YOLO11l. Lo script usa ora
YOLO11n per default e deve essere rieseguito sul contratto corrente.

## Nota storica

La prima versione della milestone usava CeyMo come unica sorgente esaustiva per
le frecce. Quel contratto unpaired è superato e i relativi smoke restano
soltanto report tecnici storici.
