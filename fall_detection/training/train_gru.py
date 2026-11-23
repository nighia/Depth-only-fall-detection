import argparse
import os
import random

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fall_detection.config import load_config
from fall_detection.data.scaling import FeatureScaler
from fall_detection.features.depth_features import FEATURE_SCHEMA_VERSION
from fall_detection.models.gru import DepthFeatureGRU
from fall_detection.schemas import STAGE_A_CLASSES


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def class_weights(y, n):
    counts = np.bincount(y, minlength=n)
    w = np.where(counts > 0, counts.sum() / (n * np.maximum(counts, 1)), 0.0)
    return torch.tensor(w, dtype=torch.float32)


def evaluate(model, x, y, loss_fn, n_classes):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = loss_fn(logits, y).item()
    pred = logits.argmax(1).numpy()
    f1 = f1_score(y.numpy(), pred, labels=list(range(n_classes)), average='macro', zero_division=0)
    return loss, float(f1)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True, help='NPZ: X_train,y_train,X_val,y_val (raw, unscaled)')
    p.add_argument('--config', default='configs/experiment_gru.yaml')
    p.add_argument('--epochs', type=int, help='override config')
    p.add_argument('--output', default='artifacts/gru_baseline.pt')
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    mc, tc, seed = cfg['model'], cfg['training'], cfg['experiment']['seed']
    epochs = a.epochs or tc['epochs']
    set_seed(seed)

    d = np.load(a.features, allow_pickle=False)
    required = {'X_train', 'y_train', 'X_val', 'y_val'}
    if not required.issubset(d.files):
        raise ValueError(f'NPZ must contain {sorted(required)}')
    n_cls = mc['num_classes']
    for k in ('y_train', 'y_val'):
        if d[k].min() < 0 or d[k].max() >= n_cls:
            raise ValueError(f'{k} has labels outside [0, {n_cls})')
    if d['X_train'].shape[-1] != mc['input_dim']:
        raise ValueError(f'input_dim {mc["input_dim"]} != data {d["X_train"].shape[-1]}')

    scaler = FeatureScaler().fit(d['X_train'])          # train only
    xtr = torch.tensor(scaler.transform(d['X_train']))
    xva = torch.tensor(scaler.transform(d['X_val']))
    ytr = torch.tensor(d['y_train'], dtype=torch.long)
    yva = torch.tensor(d['y_val'], dtype=torch.long)

    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=tc['batch_size'], shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    model = DepthFeatureGRU(mc['input_dim'], mc['hidden_dim'], mc['num_layers'],
                            n_cls, mc['dropout'])
    opt = torch.optim.AdamW(model.parameters(), lr=tc['learning_rate'],
                            weight_decay=tc.get('weight_decay', 1e-4))
    weights = class_weights(d['y_train'], n_cls)
    print('class weights (train):', weights.tolist())
    train_loss_fn = nn.CrossEntropyLoss(weight=weights)
    val_loss_fn = nn.CrossEntropyLoss()

    best, stale = float('inf'), 0
    for epoch in range(epochs):
        model.train(); total = 0.0
        for xb, yb in loader:
            opt.zero_grad(); loss = train_loss_fn(model(xb), yb)
            loss.backward(); opt.step(); total += loss.item() * len(xb)
        val, f1 = evaluate(model, xva, yva, val_loss_fn, n_cls)
        print(f'epoch {epoch+1}: train={total/len(xtr):.4f} val={val:.4f} val_macro_f1={f1:.3f}')
        if val < best:
            best, stale = val, 0
            os.makedirs(os.path.dirname(a.output) or '.', exist_ok=True)
            torch.save({'model_state': model.state_dict(), 'model_cfg': mc,
                        'scaler': scaler.state_dict(), 'classes': STAGE_A_CLASSES,
                        'feature_schema_version': FEATURE_SCHEMA_VERSION,
                        'experiment': cfg['experiment'], 'best_val_loss': best}, a.output)
        else:
            stale += 1
            if stale >= tc['patience']:
                print(f'early stop at epoch {epoch+1}'); break


if __name__ == '__main__':
    main()
