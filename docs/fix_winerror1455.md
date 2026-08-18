# Fix: WinError 1455 — "Il file di paging è troppo piccolo"

## Sintomo

Durante il training YOLO (soprattutto su Windows) il processo crasha con:

```
OSError: [WinError 1455] Il file di paging è troppo piccolo per essere completata l'operazione.
Error loading "...\torch\lib\cublas64_13.dll" or one of its dependencies.
```

**Non è un problema di CUDA OOM** (quello ha un messaggio diverso) e **non riguarda i dati/labels**. È il commit charge totale (RAM fisica + pagefile) che supera il file di paging di Windows.

## Causa

La catena tipica:

1. Un run parte con un batch troppo grande per la VRAM → **CUDA OOM**.
2. Ultralytics fa auto-retry a un batch più piccolo (`Reducing to batch=10 and retrying`).
3. Al retry vengono spawnati **N processi dataloader worker**. Su Windows ogni worker usa lo start-method `spawn`, quindi **re-importa l'intera libreria torch** e **ricarica `cublas64_*.dll` (DLL cuBLAS, grande)** per ciascun processo.
4. Commit charge totale (main + N worker + allocazioni GPU) > RAM fisica + pagefile → Windows rifiuta di caricare la DLL → **WinError 1455**.

## Fix di root: ingrandire il file di paging (pagefile)

Operazione di sistema, **una tantum**, da fare come amministratore:

1. Apri il menu Start e cerca **"Visualizza impostazioni di sistema avanzate"**
   (oppure: *Impostazioni → Sistema → Info → Impostazioni di sistema avanzate*).
2. Nella finestra **Proprietà del sistema**, tab **Avanzate**, sezione **Prestazioni** → clicca **Impostazioni…**.
3. Tab **Avanzate** → sezione **Memoria virtuale** → clicca **Cambia…**.
4. **Deseleziona** "Gestione automatica delle dimensioni del file di paging per tutte le unità".
5. Seleziona il disco con più spazio libero (di solito `C:`) → spunta **Dimensioni personalizzate**:
   - **Dimensioni iniziali (MB):** `24576` (24 GB)
   - **Dimensioni massime (MB):** `49152` (48 GB)

   Regola empirica: **RAM fisica + pagefile ≥ ~64 GB** per training YOLO con worker multipli su dataset grandi.
6. Clicca **Imposta** → **OK** su tutte le finestre.
7. **Riavvia il PC** (obbligatorio: il pagefile nuovo viene applicato solo al boot).

## Verifica post-riavvio

Apri **Task Manager** → scheda **Prestazioni** → **Memoria**:
- controlla "Memoria virtuale / file di paging" in uso e il limite;
- oppure scheda **Dettagli**, aggiungi la colonna "Commit (byte)".

Deve esserci margine ampio (non al 99%).

Oppure da terminale (PowerShell):

```powershell
Get-CimInstance Win32_PageFileUsage | Select-Object Name, AllocatedBaseSize, CurrentUsage
```

`AllocatedBaseSize` è in MB e deve riflettere il nuovo valore (~24000 o più).

## Fix complementari nel codice (già applicati)

Oltre al pagefile, il repo ora riduce il rischio di ribattere nell'errore:

1. **`hyp_base.yaml`: `workers: 4`** (era 8). Ogni worker in meno = una cuBLAS in meno da caricare via spawn. Abbassa ulteriormente con `--workers 2` se il problema persiste.
2. **Baseline `train.py`: il modello e la scala sono espliciti.** La baseline
   ATLAS usa YOLOv8/YOLO26 senza P2 e stampa modello, input, batch e workers
   all'avvio. La mainline TLR-YOLO-MTL usa invece
   `configs/tlr_yolo_mtl_train.yaml`, con `workers=2` e `prefetch_factor=1`.
3. **Per la baseline usa il batch automatico ufficiale** (`batch=-1`) per YOLOv8n a 1280; se
   compare CUDA OOM, passa un batch esplicito più piccolo o riduci `--imgsz`.
