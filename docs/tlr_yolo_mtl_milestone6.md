# TLR-YOLO-MTL — Milestone 6: contesto frecce e FiLM

**Stato corrente:** implementato e abilitabile sui dati DTLD paired.

## Percorso differenziabile

La relevance usa le predizioni dense P3-P5 della testa frecce prima della NMS.
Per ogni livello e direzione costruisce:

```text
sigmoid(arrow score) * sigmoid(direction logit)
```

P4 e P5 sono interpolate alla risoluzione P3. Una convoluzione 1x1 fonde i
livelli in tre heatmap direzionali; pooling 4x8 e MLP producono `g_arrow` da 64
dimensioni.

L'embedding genera `gamma` e `beta` FiLM per ciascun livello della branch
relevance. Le feature locali includono anche coordinate normalizzate,
dimensione della box prevista e probabilità del pittogramma; lo stato non è
fornito alla relevance. FiLM e proiezioni sono inizializzati a zero, quindi il
modello parte dal comportamento locale del Milestone 5.

Non sono presenti hard matching, soglie o NMS nel percorso di training. Tutta
l'architettura usa P3, P4 e P5; P2 resta esclusa.

## Gradiente controllato sul paired DTLD

Il corpus contiene 28.525 immagini con relevance e annotazioni frecce
esaustive sulla stessa immagine. Per questi record:

```text
Q_controlled = stopgrad(Q) + 0,25 * (Q - stopgrad(Q))
```

Su ATLAS, LISA e DTLD official-test la scala è zero. Il context encoder e FiLM
possono apprendere dal forward delle heatmap, ma la loss relevance non deforma
la testa frecce sui record senza supervisione paired.

## Ablazioni

Il contributo contestuale deve essere misurato con tre configurazioni:

1. `local-only`: FiLM disabilitato;
2. `context-stopgrad`: FiLM abilitato e scala gradiente zero;
3. `paired-context`: FiLM abilitato e scala 0,25 sui record paired.

## Verifica

```powershell
.\.venv\Scripts\python.exe -B -m scripts.check_tlr_yolo_mtl_context
```

Il controllo usa il manifest `datasets/tlr_mtl_dtld_paired/records.jsonl`,
verifica heatmap, embedding, forward FiLM e gradiente nullo/non nullo alle scale
0 e 0,25. Report: `results/tlr_yolo_mtl/milestone6_context.json`.

Il report attualmente salvato usa YOLO11l e il vecchio manifest unpaired da
119.925 immagini. Deve essere sovrascritto rieseguendo lo script corrente prima
del training finale.

## Nota storica

La versione precedente bloccava sempre il gradiente perché DTLD e CeyMo erano
unpaired. Tale limitazione è stata rimossa dal Milestone 10; CeyMo non appartiene
più alla pipeline corrente.
