# TLR-YOLO-MTL — Milestone 8: valutazione e calibrazione

**Stato corrente:** primitive metriche implementate; inferenza end-to-end,
selezione di `best.pt` e risultati finali non ancora prodotti. La metodologia
canonica è in `metodologia_pipeline_attuale.md`.

## Metriche implementate

Il modulo `tlr_yolo_mtl/evaluation` fornisce primitive deterministiche e senza
dipendenze da framework metrici esterni per:

- matching prediction–ground truth score-ordered a IoU ≥ 0,5;
- AP, AUROC e AUPRC binarie;
- precision, recall, F1, balanced accuracy, Brier score ed ECE;
- confusion matrix, accuracy, balanced accuracy e macro-F1 multi-classe;
- macro-F1 e exact-match accuracy multi-label per le frecce;
- selezione della soglia con un vincolo minimo di recall;
- temperature scaling separato per logits binari o multi-classe;
- score composito per la scelta del checkpoint.

Lo score validation è quello definito nel piano:

```text
0,25 × AP_TL_tiny
+ 0,15 × macro-F1_state
+ 0,15 × macro-F1_pictogram
+ 0,15 × AP_arrow
+ 0,30 × AUPRC_relevance
```

La calibrazione ignora target `-1` e cerca una temperatura scalare positiva che
non peggiori la negative log-likelihood. Detection score, relevance score e
joint score restano quantità distinte anche nella selezione delle soglie.

## Protocollo sperimentale corrente

Il confronto minimo per misurare il contributo delle frecce deve mantenere
invariati split, sampler, seed e budget di step:

1. relevance locale senza contesto;
2. contesto abilitato con stop-gradient verso la testa frecce;
3. contesto paired con scala gradiente 0,25.

La validation serve per checkpoint, soglie e temperature. Il test viene usato
una sola volta dopo il congelamento del protocollo. DTLD official-test possiede
relevance ma non ground truth frecce: per una valutazione paired completa serve
annotarlo oppure congelare prima un holdout paired per sequenza.

## Test

I test sintetici verificano:

- unicità dei match e conservazione degli indici prediction;
- comportamento di AP e confusion matrix;
- direzioni multi-label composte;
- soglia con recall vincolato;
- formula dello score validation;
- NLL non crescente dopo temperature fitting.

## Stato

Implementazione delle metriche completata. Il Milestone 8 sperimentale resta
aperto: confusion matrix, curve, calibrazione e risultati per dimensione/
occlusione richiedono checkpoint addestrati e inferenza sulla validation. Il
test finale non viene anticipato né usato per scegliere soglie.
