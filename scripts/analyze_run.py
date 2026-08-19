import json
import numpy as np
import os

metrics_path = 'runs/tlr_yolo_mtl_single_phase_seed42/metrics.jsonl'

if not os.path.exists(metrics_path):
    print("Metrics file not found:", metrics_path)
    exit(1)

with open(metrics_path, 'r') as f:
    lines = [json.loads(line.strip()) for line in f if line.strip()]

train_steps = [l for l in lines if 'event' not in l]
val_epochs = [l for l in lines if l.get('event') == 'val']

print("--- LEARNING RATE EVOLUTION ---")
for ep in range(0, 46, 5):
    ep_steps = [l for l in train_steps if l.get('epoch') == ep and 'learning_rates' in l]
    if ep_steps:
        bb_lr = ep_steps[0]['learning_rates'][0]
        hd_lr = ep_steps[0]['learning_rates'][2]
        print(f"Epoch {ep:2d}: Backbone LR = {bb_lr:.6e} | Head LR = {hd_lr:.6e}")


train_by_epoch = {}
for s in train_steps:
    ep = s['epoch']
    if ep not in train_by_epoch:
        train_by_epoch[ep] = {
            'total_loss': [], 'det_loss': [], 'state_loss': [],
            'rel_loss': [], 'nwd_loss': [], 'round_loss': [], 'maneuver_loss': []
        }
    comps = s.get('mean_loss_components', {})
    train_by_epoch[ep]['total_loss'].append(comps.get('total', s.get('mean_micro_loss', 0)))
    train_by_epoch[ep]['det_loss'].append(comps.get('detection', 0))
    train_by_epoch[ep]['state_loss'].append(comps.get('state', 0))
    train_by_epoch[ep]['rel_loss'].append(comps.get('relevance', 0))
    train_by_epoch[ep]['nwd_loss'].append(comps.get('nwd', 0))
    train_by_epoch[ep]['round_loss'].append(comps.get('round', 0))
    train_by_epoch[ep]['maneuver_loss'].append(comps.get('maneuver', 0))

train_summary = {}
for ep, vals in train_by_epoch.items():
    train_summary[ep] = {k: float(np.mean(v)) for k, v in vals.items()}

print(f"Total completed val epochs: {len(val_epochs)}")
print(f"Total train epochs logged: {len(train_summary)}")

print("\n" + "="*120)
print(f"{'Ep':>3} | {'TrLoss':>7} | {'ValLoss':>7} | {'SelScore':>8} | {'mAP50':>6} | {'mAP50-95':>8} | {'AP_TL50':>7} | {'AP_Arr50':>8} | {'Rel_F1':>6} | {'Rel_AUC':>7} | {'St_Acc':>6} | {'St_mF1':>6} | {'Best?':>5}")
print("-" * 120)

all_records = []
for v in val_epochs:
    ep = v['epoch']
    tr = train_summary.get(ep, {})
    tr_loss = tr.get('total_loss', float('nan'))
    v_loss = v['mean_losses']['total']
    sel = v['selection_score']
    m50 = v['detection']['map50']
    m50_95 = v['detection']['map50_95']
    ap_tl = v['detection']['ap_tl_50']
    ap_arr = v['detection']['ap_arrow_50']
    rel_f1 = v['relevance']['f1']
    rel_auprc = v['relevance']['auprc']
    st_acc = v['attributes']['state_accuracy']
    st_mf1 = v['attributes']['state_macro_f1']
    is_best = '*** BEST ***' if v.get('is_best') else ''
    
    all_records.append({
        'epoch': ep,
        'tr_loss': tr_loss,
        'val_loss': v_loss,
        'sel_score': sel,
        'm50': m50,
        'm50_95': m50_95,
        'ap_tl': ap_tl,
        'ap_arr': ap_arr,
        'rel_f1': rel_f1,
        'rel_auprc': rel_auprc,
        'st_acc': st_acc,
        'st_mf1': st_mf1,
        'is_best': v.get('is_best', False),
        'tr_det_loss': tr.get('det_loss', float('nan')),
        'val_det_loss': v['mean_losses'].get('detection', float('nan')),
        'tr_rel_loss': tr.get('rel_loss', float('nan')),
        'val_rel_loss': v['mean_losses'].get('relevance', float('nan')),
        'tr_state_loss': tr.get('state_loss', float('nan')),
        'val_state_loss': v['mean_losses'].get('state', float('nan')),
    })
    
    flag = " [BEST]" if v.get('is_best') else ""
    print(f"{ep:3d} | {tr_loss:7.3f} | {v_loss:7.3f} | {sel:8.4f} | {m50:6.3f} | {m50_95:8.4f} | {ap_tl:7.4f} | {ap_arr:8.4f} | {rel_f1:6.3f} | {rel_auprc:7.4f} | {st_acc:6.3f} | {st_mf1:6.3f} | {flag}")

print("="*120)

print("\n" + "="*100)
print("LOSS COMPONENT DECOMPOSITION (Train vs Val)")
print(f"{'Ep':>3} | {'TrDet':>7} {'ValDet':>7} | {'TrState':>7} {'ValState':>8} | {'TrRel':>7} {'ValRel':>7} | {'TrNWD':>7} {'ValNWD':>7}")
print("-" * 100)
for r in all_records:
    ep = r['epoch']
    print(f"{ep:3d} | {r['tr_det_loss']:7.3f} {r['val_det_loss']:7.3f} | {r['tr_state_loss']:7.4f} {r['val_state_loss']:8.4f} | {r['tr_rel_loss']:7.4f} {r['val_rel_loss']:7.4f} | {r['tr_loss']-r['tr_det_loss']:7.4f} {r['val_loss']-r['val_det_loss']:7.4f}")

# Best values analysis
best_sel = max(all_records, key=lambda x: x['sel_score'])
best_m50 = max(all_records, key=lambda x: x['m50'])
best_m50_95 = max(all_records, key=lambda x: x['m50_95'])
best_ap_tl = max(all_records, key=lambda x: x['ap_tl'])
best_rel_f1 = max(all_records, key=lambda x: x['rel_f1'])
best_rel_auc = max(all_records, key=lambda x: x['rel_auprc'])
best_st_acc = max(all_records, key=lambda x: x['st_acc'])
best_st_mf1 = max(all_records, key=lambda x: x['st_mf1'])
min_val_loss = min(all_records, key=lambda x: x['val_loss'])

print("\n--- BEST METRICS RECORDED ---")
print(f"Best Selection Score : {best_sel['sel_score']:.4f} at Epoch {best_sel['epoch']}")
print(f"Best Val Loss        : {min_val_loss['val_loss']:.4f} at Epoch {min_val_loss['epoch']}")
print(f"Best mAP50           : {best_m50['m50']:.4f} at Epoch {best_m50['epoch']}")
print(f"Best mAP50-95        : {best_m50_95['m50_95']:.4f} at Epoch {best_m50_95['epoch']}")
print(f"Best AP_TL50         : {best_ap_tl['ap_tl']:.4f} at Epoch {best_ap_tl['epoch']}")
print(f"Best Relevance F1    : {best_rel_f1['rel_f1']:.4f} at Epoch {best_rel_f1['epoch']}")
print(f"Best Relevance AUPRC : {best_rel_auc['rel_auprc']:.4f} at Epoch {best_rel_auc['epoch']}")
print(f"Best State Acc       : {best_st_acc['st_acc']:.4f} at Epoch {best_st_acc['epoch']}")
print(f"Best State Macro F1  : {best_st_mf1['st_mf1']:.4f} at Epoch {best_st_mf1['epoch']}")

# Recent trends (last 10 epochs vs previous)
if len(all_records) >= 10:
    last_10 = all_records[-10:]
    prev_10 = all_records[-20:-10] if len(all_records) >= 20 else all_records[:10]
    
    print("\n--- TREND ANALYSIS (Last 10 vs Previous 10 epochs) ---")
    print(f"Train Loss avg: {np.mean([r['tr_loss'] for r in prev_10]):.4f} -> {np.mean([r['tr_loss'] for r in last_10]):.4f}")
    print(f"Val Loss avg  : {np.mean([r['val_loss'] for r in prev_10]):.4f} -> {np.mean([r['val_loss'] for r in last_10]):.4f}")
    print(f"Sel Score avg : {np.mean([r['sel_score'] for r in prev_10]):.4f} -> {np.mean([r['sel_score'] for r in last_10]):.4f}")
    print(f"mAP50 avg     : {np.mean([r['m50'] for r in prev_10]):.4f} -> {np.mean([r['m50'] for r in last_10]):.4f}")
    print(f"mAP50-95 avg  : {np.mean([r['m50_95'] for r in prev_10]):.4f} -> {np.mean([r['m50_95'] for r in last_10]):.4f}")
    print(f"Rel F1 avg    : {np.mean([r['rel_f1'] for r in prev_10]):.4f} -> {np.mean([r['rel_f1'] for r in last_10]):.4f}")
    print(f"State Acc avg : {np.mean([r['st_acc'] for r in prev_10]):.4f} -> {np.mean([r['st_acc'] for r in last_10]):.4f}")

# Overfitting check details:
epochs_since_best_sel = all_records[-1]['epoch'] - best_sel['epoch']
epochs_since_best_val_loss = all_records[-1]['epoch'] - min_val_loss['epoch']
epochs_since_best_m50 = all_records[-1]['epoch'] - best_m50['epoch']
epochs_since_best_m50_95 = all_records[-1]['epoch'] - best_m50_95['epoch']

print("\n--- OVERFITTING / STOPPING CRITERIA CHECK ---")
print(f"Current Epoch: {all_records[-1]['epoch']}")
print(f"Epochs since Best Selection Score: {epochs_since_best_sel}")
print(f"Epochs since Best Val Loss: {epochs_since_best_val_loss}")
print(f"Epochs since Best mAP50: {epochs_since_best_m50}")
print(f"Epochs since Best mAP50-95: {epochs_since_best_m50_95}")

