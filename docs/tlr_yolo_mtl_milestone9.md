# TLR-YOLO-MTL — Milestone 9: export e deployment

**Stato corrente:** export architetturale verificato; artefatto finale da
rigenerare dopo il training da `best.pt`. La metodologia canonica è in
`metodologia_pipeline_attuale.md`.

## Output e post-processing

Il grafo completo espone sei tensori fissi:

1. traffic-light detection `[1, 5, 26250]`;
2. state logits `[1, 4, 26250]`;
3. pictogram logits `[1, 4, 26250]`;
4. arrow detection `[1, 5, 26250]`;
5. arrow direction logits `[1, 3, 26250]`;
6. relevance logits `[1, 1, 26250]`.

Il post-processing usa due NMS distinte, massimo 100 semafori e 50 frecce, e
conserva l'indice denso originario di ogni detection. Gli indici raccolgono
stato, pittogramma, rilevanza e direzione dalla stessa candidate. Non vengono
creati oggetti aggiuntivi né ricostruiti semafori occlusi.

## ONNX — verifica storica da rigenerare

Il report salvato documenta un export YOLO11l a batch 1, shape
`[1, 3, 800, 1600]`, opset 17 e FP16:

- dimensione: 54.490.733 byte;
- nodi: 1.253;
- checker ONNX: superato;
- livelli: soltanto P3–P5;
- sei output con shape attese.

La parità PyTorch–ONNX Runtime è stata misurata in FP32 a 320×320 sul provider
CPU. Tutti gli output rispettano `atol=0,002`; il massimo errore assoluto
osservato è 5,65×10⁻⁴.

## Profiling preliminare storico

PyTorch CUDA FP16, batch 1, 800×1600, 20 warm-up e 100 iterazioni:

- media: 22,90 ms;
- mediana: 22,71 ms;
- p90: 23,40 ms;
- p95: 23,85 ms;
- picco memoria: 282.231.296 byte.

È solo latenza rete: non include trasferimento input, preprocessing, NMS o
post-processing e non sostituisce la misura TensorRT end-to-end richiesta.

## Comando

```powershell
.\.venv\Scripts\python.exe -B -m scripts.check_tlr_yolo_mtl_deployment
```

Artefatti:

- `results/tlr_yolo_mtl/milestone9_deployment.json`: report storico presente;
- `results/tlr_yolo_mtl/tlr-yolo-mtl-p3-p5-fp16.onnx`: output riproducibile,
  attualmente non conservato nel repository.

## Stato

Export dell'architettura, parità ONNX e NMS allineata sono stati dimostrati sul
prototipo YOLO11l. Lo script corrente usa YOLO11n per default e deve essere
rieseguito per aggiornare report e binario della mainline. Dopo il training,
l'export deve essere rigenerato da `best.pt`. TensorRT non è installato
nell'ambiente attuale, quindi build engine, parità FP16 TensorRT e latenza
end-to-end su almeno 2.000 immagini restano aperte.
