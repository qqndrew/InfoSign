"""Joint CBM with individually supervised, heterogeneous concept items."""
import copy
from pathlib import Path
import pandas as pd
from infosign.common import read_json, write_json, save_torch, backbone_dir
from infosign.train import load
from datetime import datetime, timezone
from functools import lru_cache
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from infosign.stats import balanced_indices, f1, threshold


def numeric(value):
    try:
        x = float(value)
        return x if np.isfinite(x) else np.nan
    except ValueError:
        return np.nan


@lru_cache(maxsize=100000)
def date_days(value):
    # Local CSVs use M/D/YY H:MM; also accept ISO exports. A clinical name
    # containing "date" is not enough: eos_dsvst=1/2 remains categorical.
    try:
        if len(value) >= 10 and value[4] == '-' and value[7] == '-':
            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
        elif value.count('/') == 2:
            date, _, time = value.partition(' ')
            year = date.rsplit('/', 1)[1]
            if len(year) not in (2, 4): return np.nan
            fmt = '%m/%d/%y' if len(year) == 2 else '%m/%d/%Y'
            if time: fmt += ' %H:%M:%S' if time.count(':') == 2 else ' %H:%M'
            dt = datetime.strptime(value, fmt)
        else: return np.nan
        if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp() / 86400
    except ValueError:
        return np.nan


def encode(values, item):
    values = np.asarray(values, str)
    if item['type'] == 'categorical':
        lookup = {v: i for i, v in enumerate(item['categories'])}
        y = np.array([lookup.get(v, -1) for v in values], dtype=np.float32)
        mask = y >= 0
    else:
        parse = date_days if item['encoding'] == 'date_days' else numeric
        y = np.array([parse(v) for v in values], dtype=np.float64)
        mask = np.isfinite(y)
        y = (y - item['mean']) / item['std']
    return np.where(mask, y, 0).astype(np.float32), mask


def prepare(train, validation, labels, schema):
    frame = labels.set_index('person_id')
    frames = [frame.reindex(list(map(str, c['person_ids']))).fillna('') for c in (train, validation)]
    x = np.asarray(train['embedding'], dtype=np.float32)
    mean, std = x.mean(0), x.std(0); std[std < 1e-8] = 1
    items, targets, masks, offset = [], [[], []], [[], []], 0
    for definition in schema['concepts']:
        item = copy.deepcopy(definition)
        values = frames[0][item['name']].to_numpy(str)
        observed = values[values != '']
        numbers = np.array([numeric(v) for v in observed])
        dates = np.array([date_days(v) for v in observed])
        categorical = bool(set(item['field_types']) & {'radio', 'checkbox', 'dropdown', 'yesno', 'truefalse'})
        item.update(mean=0., std=1., categories=[], encoding='raw_category')
        if len(observed) and not categorical and (np.isfinite(numbers).all() or np.isfinite(dates).all()):
            use_dates = not np.isfinite(numbers).all()
            y = dates if use_dates else numbers
            item.update(type='continuous', encoding='date_days' if use_dates else 'number',
                        mean=float(y.mean()), std=float(y.std()) if y.std() > 1e-8 else 1.)
            levels = len(np.unique(y)); width = 1
        else:
            categories = sorted(set(observed))
            item.update(type='categorical', categories=categories)
            levels = len(categories); width = max(1, levels)
        item.update(start=offset, stop=offset + width, n_train_observed=len(observed),
                    learnable=levels > 1, status='learned' if levels > 1 else 'constant' if levels else 'no_training_label')
        offset += width
        for k, f in enumerate(frames):
            y, present = encode(f[item['name']].to_numpy(str), item)
            targets[k].append(y); masks[k].append(present)
        items.append(item)
    if not any(c['learnable'] for c in items): raise ValueError('No varying concept labels in the training cohort')
    spec = {'items': items, 'width': offset, 'input_dim': x.shape[1], 'x_mean': mean.tolist(), 'x_std': std.tolist()}
    data = []
    for k, cache in enumerate((train, validation)):
        data.append({'x': torch.tensor((np.asarray(cache['embedding']) - mean) / std, dtype=torch.float32),
                     'y': torch.as_tensor(cache['y'], dtype=torch.float32),
                     'targets': torch.tensor(np.stack(targets[k], axis=1)),
                     'mask': torch.tensor(np.stack(masks[k], axis=1))})
    return spec, data


class IndividualCBM(nn.Module):
    def __init__(self, spec, hidden=32):
        super().__init__(); self.spec = spec
        self.concept_layer = nn.Linear(spec['input_dim'], spec['width'])
        self.task_head = nn.Sequential(nn.Linear(spec['width'], hidden), nn.ReLU(), nn.Linear(hidden, 1))
        active = np.zeros(spec['width'], np.float32); constant = active.copy()
        numeric_columns, numeric_items, categories = [], [], {}
        for j, c in enumerate(spec['items']):
            sl = slice(c['start'], c['stop'])
            if c['learnable']:
                active[sl] = 1
                if c['type'] == 'continuous': numeric_columns.append(c['start']); numeric_items.append(j)
                else: categories.setdefault(c['stop'] - c['start'], []).append(j)
            elif c['type'] == 'categorical' and c['categories']:
                constant[c['start']] = 1
        self.register_buffer('active', torch.tensor(active), persistent=False)
        self.register_buffer('constant', torch.tensor(constant), persistent=False)
        self.register_buffer('num_columns', torch.tensor(numeric_columns, dtype=torch.long), persistent=False)
        self.register_buffer('num_items', torch.tensor(numeric_items, dtype=torch.long), persistent=False)
        self.widths = sorted(categories)
        for width, js in categories.items():
            columns = [i for j in js for i in range(spec['items'][j]['start'], spec['items'][j]['stop'])]
            self.register_buffer(f'cat_columns_{width}', torch.tensor(columns), persistent=False)
            self.register_buffer(f'cat_items_{width}', torch.tensor(js), persistent=False)

    def bottleneck(self, x):
        raw = self.concept_layer(x)
        z = raw * self.active + self.constant
        for width in self.widths:
            cols = getattr(self, f'cat_columns_{width}')
            probs = raw[:, cols].reshape(len(x), -1, width).softmax(-1).flatten(1)
            z = z.index_copy(1, cols, probs)
        return raw, z

    def forward(self, x):
        raw, z = self.bottleneck(x)
        return raw, z, self.task_head(z).squeeze(-1)

    def losses(self, data):
        raw, z, logits = self(data['x'])
        losses = raw.new_full((len(self.spec['items']),), float('nan'))
        if self.num_items.numel():
            js = self.num_items; mask = data['mask'][:, js]
            per = ((z[:, self.num_columns] - data['targets'][:, js]).square() * mask).sum(0) / mask.sum(0).clamp_min(1)
            losses = losses.index_copy(0, js, torch.where(mask.any(0), per, float('nan')))
        for width in self.widths:
            js = getattr(self, f'cat_items_{width}'); cols = getattr(self, f'cat_columns_{width}')
            log_probs = raw[:, cols].reshape(len(raw), -1, width).log_softmax(-1)
            ce = -log_probs.gather(2, data['targets'][:, js].long().unsqueeze(-1)).squeeze(-1)
            mask = data['mask'][:, js]
            per = (ce * mask).sum(0) / mask.sum(0).clamp_min(1)
            losses = losses.index_copy(0, js, torch.where(mask.any(0), per, float('nan')))
        return losses, F.binary_cross_entropy_with_logits(logits, data['y'])


def state(model): return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def fit(train, val, labels, schema, cfg, device, masked_train=None, model_cls=None):
    """Frozen backbone; concept warm-up, head fitting, then joint readout tuning.

    Mixed mode oversamples participants once per epoch, then presents both
    views of each sampled participant with equal weight. Encoding statistics
    and vocabulary are fitted to unique original training participants only.
    A model with a bypass module trains that module with the outcome head.
    """
    torch.manual_seed(cfg['seed'])
    spec, data = prepare(train, val, labels, schema)
    data = [{k: v.to(device) for k, v in d.items()} for d in data]
    masked = None
    if masked_train is not None:
        masked = torch.as_tensor((np.asarray(masked_train)-np.asarray(spec['x_mean']))/np.asarray(spec['x_std']),
                                 dtype=torch.float32, device=device)
    model = (model_cls or IndividualCBM)(spec).to(device); history = []; selections = {}; reference = None
    phases = (('concept', cfg['concept_epochs']), ('head', cfg['head_epochs']), ('joint', cfg['joint_epochs']))
    for phase_index, (phase, epochs) in enumerate(phases):
        model.concept_layer.requires_grad_(phase != 'head'); model.task_head.requires_grad_(phase != 'concept')
        bypass = getattr(model, 'bypass', None)
        if bypass is not None: bypass.requires_grad_(phase != 'concept')
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                     lr=cfg['cbm_learning_rate'], weight_decay=cfg['cbm_weight_decay'])
        best, best_key, best_epoch, stale = state(model), (-float('inf'), -float('inf')), 0, 0
        for epoch in range(epochs+1):
            if epoch:
                indices = balanced_indices(train['y'], cfg['seed'] + phase_index*100000 + epoch)
                idx = torch.as_tensor(indices, device=device)
                batch = {k: v[idx] for k, v in data[0].items()}
                if masked is not None:
                    batch = {k: torch.cat((v, masked[idx] if k == 'x' else v)) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                c, loss_y = model.losses(batch)
                objective = c.nanmean() if phase == 'concept' else loss_y if phase == 'head' else loss_y + cfg['concept_lambda']*c.nanmean()
                objective.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
            with torch.no_grad():
                c, loss_y = model.losses(data[1])
                p = model(data[1]['x'])[2].sigmoid().cpu().numpy()
            per = c.cpu().numpy(); concept_loss = float(c.nanmean())
            known = np.isfinite(per)
            eligible = phase != 'joint' or bool(np.all(per[known] <= reference[known]*(1+cfg['concept_tolerance'])+1e-8))
            cut = threshold(val['y'], p); score = float(f1(val['y'], p, cut))
            key = (-concept_loss, -float(loss_y)) if phase == 'concept' else (score, -float(loss_y))
            history.append(dict(phase=phase, epoch=epoch, val_f1=score, threshold=cut,
                                val_task_loss=float(loss_y), val_concept_loss=concept_loss, eligible=eligible))
            if eligible and key > best_key:
                best, best_key, best_epoch, stale = state(model), key, epoch, 0
            else: stale += 1
            if epoch % 25 == 0 or stale >= cfg['concept_patience']:
                print(f'[CBM {phase}] epoch={epoch} val_F1={score:.4f} concept_loss={concept_loss:.4f} eligible={eligible}', flush=True)
            if stale >= cfg['concept_patience']: break
        model.load_state_dict(best)
        selections[phase] = dict(epoch=best_epoch, status='sequential_fallback' if phase=='joint' and best_epoch==0 else 'fitted')
        if phase == 'head':
            with torch.no_grad(): reference = model.losses(data[1])[0].cpu().numpy()
    _, p = predict(model, val['embedding'])
    artifact = dict(spec=spec, states={'joint':state(model)}, selection=selections,
                    threshold=threshold(val['y'], p), backbone_frozen=True,
                    training_views=['full','masked'] if masked_train is not None else ['basic'],
                    concept_lambda=cfg['concept_lambda'], concept_tolerance=cfg['concept_tolerance'])
    if getattr(model, 'bypass_kind', None): artifact['bypass'] = model.bypass_kind
    return artifact, history


def restore(artifact, device, arm='joint'):
    model = IndividualCBM(artifact['spec']).to(device)
    model.load_state_dict(artifact['states'][arm]); model.requires_grad_(False)
    return model.eval()


@torch.no_grad()
def predict(model, embedding):
    x = (np.asarray(embedding) - np.asarray(model.spec['x_mean'])) / np.asarray(model.spec['x_std'])
    _, z, logits = model(torch.tensor(x, dtype=torch.float32, device=next(model.parameters()).device))
    return z, logits.sigmoid().cpu().numpy()


def run_cbm(task, cfg, device):
    task_root = Path(cfg['output_root'])/task
    root = task_root/'mixed_cbm'; root.mkdir(parents=True, exist_ok=True)
    if (root/'DONE.json').exists(): print(f'[reuse] {task}/mixed_cbm', flush=True); return
    source = backbone_dir(task,cfg)/'full'
    train = load(source/'features_train.pt'); val = load(source/'features_val.pt')
    labels = pd.read_csv(Path(cfg['output_root'])/'labels'/task/'labels.csv', dtype=str, keep_default_na=False)
    schema = read_json(Path(cfg['output_root'])/'labels'/task/'schema.json')
    validation = {**val, 'embedding':val['masked_embedding']}
    torch.set_float32_matmul_precision('highest')
    artifact, history = fit(train, validation, labels, schema, cfg, device,
                            masked_train=train['masked_embedding'])
    save_torch(root/'model.pt', artifact)
    pd.DataFrame(history).to_csv(root/'training.csv', index=False)
    write_json(root/'concept_definitions.json', artifact['spec']['items'])
    write_json(root/'DONE.json', dict(task=task, mode='mixed', threshold=artifact['threshold'],
        selection=artifact['selection'], inventory_ids=schema['inventory_ids'],
        learned_items=sum(c['learnable'] for c in artifact['spec']['items']),
        input_backbone='full', backbone_frozen=True))


def concept_id_groups(items):
    groups = {}
    for c in items:
        cid = str(c['concept_id'])
        g = groups.setdefault(cid, dict(concept_id=cid, concept_name=c['concept_name'],
            source_fields=[], columns=[], learnable=False))
        if c['source_field'] not in g['source_fields']: g['source_fields'].append(c['source_field'])
        g['columns'].extend(range(c['start'], c['stop']))
        g['learnable'] |= c['learnable']
    return groups


@torch.no_grad()
def restoration_batch(model, missing, full, columns):
    """Restore a whole ID efficiently through the additive first head layer."""
    first, _, last = model.task_head; base = first(missing)
    values = []
    for cols in columns:
        if len(cols) == model.spec['width']:
            values.append(model.task_head(full).squeeze(-1).sigmoid())
            continue
        delta = (full[:,cols]-missing[:,cols]) @ first.weight[:,cols].T
        values.append(last(torch.relu(base+delta)).squeeze(-1).sigmoid())
    return torch.stack(values).cpu().numpy()
