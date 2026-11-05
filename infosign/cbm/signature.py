"""Two selected-concept Mixed CBMs from one validation concept list.

The bottleneck run routes the outcome through the selected concepts only.
The residual run also passes the embedding component orthogonal to that layer.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from infosign.common import TASK_LABELS, backbone_dir, read_json, write_json, save_torch
from infosign.report import interval, md
from infosign.stats import PairedF1, auroc_selection
from infosign.train import load
from infosign.cbm.diagnostics import METRICS, interval_fields
from infosign.cbm.model import IndividualCBM, fit, predict, restore


VERSION = 3
BYPASS = 'residual'
SELECTION_SIGMAS = 1
CUTOFF_LABEL = f'μ + {SELECTION_SIGMAS:g}σ'


def validation_concepts(restoration):
    """Select Concept IDs from the validation predicted-restoration distribution.

    Test rows are removed before the mean and population SD are computed.
    An ID is kept only when its gain is strictly above μ + 1σ.
    """
    group = restoration[
        (restoration['split'].astype(str) == 'val')
        & (restoration['source'].astype(str) == 'full_input_prediction')
        & (restoration['population'].astype(str) == 'all_participants')
    ].copy()
    if group.empty:
        raise ValueError('Validation predicted-restoration rows are missing')
    group = group.reset_index(drop=True)
    gains = pd.to_numeric(group['auroc_gain'], errors='coerce').to_numpy(float)
    for key, values in auroc_selection(gains, sigmas=SELECTION_SIGMAS).items():
        group[key] = values
    selected = group[group['auroc_selected'].astype(bool)].sort_values(
        ['auroc_rank_within_split', 'concept_id'], kind='stable')
    selected = selected.drop_duplicates('concept_id', keep='first')
    first = group.iloc[0]
    concepts = [dict(
        concept_id=str(row.concept_id),
        concept_name=str(row.concept_name),
        source_fields=str(row.source_fields) if 'source_fields' in group.columns else '',
        validation_auroc_gain=float(row.auroc_gain),
        validation_rank=int(row.auroc_rank_within_split),
    ) for row in selected.itertuples(index=False)]
    if not np.isfinite(first.auroc_selection_mean):
        raise ValueError('Validation predicted-restoration gains are not finite')
    return dict(
        split='val', source='full_input_prediction', population='all_participants',
        n_evaluated=int(first.auroc_selection_n),
        mean=float(first.auroc_selection_mean),
        std=float(first.auroc_selection_std),
        sigmas=SELECTION_SIGMAS,
        cutoff=float(first.auroc_selection_cutoff),
        concept_ids=[item['concept_id'] for item in concepts],
        concepts=concepts)


class ResidualSignatureCBM(IndividualCBM):
    """Selected concepts plus the embedding component those concepts do not read."""

    bypass_kind = BYPASS

    def __init__(self, spec, hidden=32):
        super().__init__(spec, hidden)
        self.bypass = torch.nn.Sequential(torch.nn.Linear(spec['input_dim'], hidden), torch.nn.ReLU())
        self.task_head = torch.nn.Sequential(
            torch.nn.Linear(spec['width'] + hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))

    def residual(self, x):
        """Remove the detached projection of x onto the concept-layer weight rows."""
        weight = self.concept_layer.weight
        with torch.no_grad():
            projected = F.linear(x, weight)
            gram = weight @ weight.transpose(0, 1)
            ridge = 1e-4 * torch.eye(gram.shape[0], device=x.device, dtype=x.dtype)
            # solve treats a matrix right-hand side as columns, so the batch axis goes last.
            coeff = torch.linalg.solve(gram + ridge, projected.transpose(0, 1)).transpose(0, 1)
            projection = coeff @ weight
        return x - projection

    def forward(self, x):
        raw, z = self.bottleneck(x)
        features = torch.cat((z, self.bypass(self.residual(x))), dim=1)
        return raw, z, self.task_head(features).squeeze(-1)


VARIANTS = (
    dict(name='bottleneck', folder='signature_cbm', model_cls=IndividualCBM, bypass=None,
         key='signature_bottleneck', label='Selected-concept CBM'),
    dict(name='residual', folder='signature_cbm_residual', model_cls=ResidualSignatureCBM,
         bypass=BYPASS, key='signature_residual', label='Selected-concept CBM with residual bypass'),
)
BASELINE_NAMES = {'mixed_cbm': 'All-concept Mixed CBM', 'full_clinical': 'Full Clinical model'}


def load_variant(artifact, device, model_cls):
    model = model_cls(artifact['spec']).to(device)
    model.load_state_dict(artifact['states']['joint'])
    model.requires_grad_(False)
    return model.eval()


def relocate_residual_checkpoint(task_root):
    """Keep an earlier residual run that was saved in the bottleneck directory."""
    source = task_root / 'signature_cbm'
    destination = task_root / 'signature_cbm_residual'
    marker = source / 'TRAINING_DONE.json'
    if destination.exists() or not marker.exists():
        return
    if read_json(marker).get('bypass') == BYPASS:
        source.rename(destination)
        print(f'[signature] moved residual checkpoint to {destination.name}', flush=True)


def restrict_schema(schema, concept_ids):
    """Keep every schema item whose Concept ID was selected, in inventory order."""
    wanted = set(map(str, concept_ids))
    concepts = [item for item in schema['concepts'] if str(item['concept_id']) in wanted]
    present = {str(item['concept_id']) for item in concepts}
    missing = sorted(wanted - present)
    if missing:
        raise ValueError('Selected Concept IDs are absent from the task schema: ' + ', '.join(missing))
    if not concepts:
        raise ValueError('No schema items belong to the selected Concept IDs')
    return {**schema, 'concepts': concepts, 'inventory_ids': len(present)}


def _probabilities(logits):
    return torch.as_tensor(logits).detach().float().sigmoid().cpu().numpy()


def _loss_draws(boot, full, masked, full_cut, masked_cut, metric):
    """Positive values mean masking nonbasic clinical inputs worsened the metric."""
    name = METRICS[metric]
    change = boot.draws(full, full_cut, name) - boot.draws(masked, masked_cut, name)
    return -change if metric == 'brier' else change


def _loss_row(boot, full, masked, full_cut, masked_cut, metric):
    draws = _loss_draws(boot, full, masked, full_cut, masked_cut, metric)
    row = boot.intervals(draws)[0]
    if metric == 'brier':
        p = boot.pvalues(masked, full, masked_cut, full_cut, METRICS[metric])
    elif metric == 'auroc':
        p = np.array([np.nan])
    else:
        p = boot.pvalues(full, masked, full_cut, masked_cut, METRICS[metric])
    row['p_value'] = float(p[0])
    return row


def evaluate(task, cfg, device, artifact, variant):
    """Score full and masked inputs and the paired loss caused by masking."""
    root = Path(cfg['output_root']) / task
    mixed_artifact = load(root / 'mixed_cbm' / 'model.pt')
    signature = load_variant(artifact, device, variant['model_cls'])
    mixed = restore(mixed_artifact, device)
    full_cut = read_json(backbone_dir(task, cfg) / 'full' / 'TRAINING_DONE.json')['threshold']
    key = variant['key']
    cuts = {key: artifact['threshold'], 'mixed_cbm': mixed_artifact['threshold'],
            'full_clinical': full_cut}
    performance, comparisons = [], []
    for split in ('val', 'test'):
        cache = load(backbone_dir(task, cfg) / 'full' / f'features_{split}.pt')
        y = cache['y'].numpy()
        boot = PairedF1(y, cfg['bootstrap_samples'], cfg['seed'], device)
        probabilities = {
            key: {
                'full': predict(signature, cache['embedding'])[1],
                'masked': predict(signature, cache['masked_embedding'])[1]},
            'mixed_cbm': {
                'full': predict(mixed, cache['embedding'])[1],
                'masked': predict(mixed, cache['masked_embedding'])[1]},
            'full_clinical': {
                'full': _probabilities(cache['original_logits']),
                'masked': _probabilities(cache['masked_logits'])}}
        for model, views in probabilities.items():
            cut = cuts[model]
            for inputs, probability in views.items():
                row = dict(task=task, split=split, model=model, inputs=inputs, threshold=cut, n=len(y))
                for metric in METRICS:
                    row.update(interval_fields(metric, boot.performance(probability, cut, METRICS[metric])))
                performance.append(row)
            for metric in METRICS:
                effect = _loss_row(boot, views['full'], views['masked'], cut, cut, metric)
                comparisons.append(dict(
                    task=task, split=split, contrast='input_loss', model=model, inputs='',
                    reference='', metric=metric, n=len(y), **effect))
        for inputs in ('full', 'masked'):
            for metric in METRICS:
                effect = boot.effects(
                    probabilities[key][inputs], probabilities['mixed_cbm'][inputs],
                    cuts[key], cuts['mixed_cbm'], METRICS[metric])[0]
                comparisons.append(dict(
                    task=task, split=split, contrast='within_input', model=key,
                    inputs=inputs, reference='mixed_cbm', metric=metric, n=len(y), **effect))
        for reference in ('mixed_cbm', 'full_clinical'):
            for metric in METRICS:
                gain = boot.intervals(
                    _loss_draws(boot, probabilities[reference]['full'], probabilities[reference]['masked'],
                                cuts[reference], cuts[reference], metric)
                    - _loss_draws(boot, probabilities[key]['full'], probabilities[key]['masked'],
                                  cuts[key], cuts[key], metric))[0]
                comparisons.append(dict(
                    task=task, split=split, contrast='robustness', model=key, inputs='',
                    reference=reference, metric=metric, n=len(y), p_value=np.nan, **gain))
        pd.DataFrame(dict(person_id=cache['person_ids'], y=y,
                          full_probability=probabilities[key]['full'],
                          masked_probability=probabilities[key]['masked'])).to_csv(
            root / variant['folder'] / f'predictions_{split}.csv', index=False)
        del boot
        print(f'[signature {task}/{variant["name"]}/{split}] full and masked inputs scored', flush=True)
    folder = root / variant['folder']
    pd.DataFrame(performance).to_csv(folder / 'performance.csv', index=False)
    pd.DataFrame(comparisons).to_csv(folder / 'comparisons.csv', index=False)


def _finished(folder):
    marker = folder / 'DONE.json'
    if not marker.exists() or read_json(marker).get('version') != VERSION:
        return False
    if read_json(marker).get('skipped'):
        return True
    return all((folder / name).exists() for name in (
        'model.pt', 'performance.csv', 'comparisons.csv', 'predictions_val.csv', 'predictions_test.csv'))


def run_variant(task, cfg, device, variant, selection):
    root = Path(cfg['output_root']) / task / variant['folder']
    root.mkdir(parents=True, exist_ok=True)
    done_path = root / 'DONE.json'
    if _finished(root):
        print(f'[reuse] {task}/{variant["folder"]}', flush=True)
        return
    write_json(root / 'selected_concepts.json', selection)
    if not selection['concept_ids']:
        write_json(done_path, dict(
            version=VERSION, task=task, variant=variant['name'], skipped=True,
            reason='no_validation_concept_above_cutoff', bypass=variant['bypass'],
            n_evaluated=selection['n_evaluated'], mean=selection['mean'], std=selection['std'],
            sigmas=selection['sigmas'], cutoff=selection['cutoff']))
        print(f'[signature {task}/{variant["name"]}] no validation concept exceeded {CUTOFF_LABEL}; skipped', flush=True)
        return
    trained_path = root / 'TRAINING_DONE.json'
    trained = read_json(trained_path) if trained_path.exists() else {}
    reuse_training = (trained.get('version') == VERSION and trained.get('bypass') == variant['bypass']
                      and trained.get('concept_ids') == selection['concept_ids'] and (root / 'model.pt').exists())
    if reuse_training:
        artifact = load(root / 'model.pt')
        print(f'[reuse training] {task}/{variant["folder"]}', flush=True)
    else:
        labels = pd.read_csv(Path(cfg['output_root']) / 'labels' / task / 'labels.csv', dtype=str, keep_default_na=False)
        schema = restrict_schema(read_json(Path(cfg['output_root']) / 'labels' / task / 'schema.json'), selection['concept_ids'])
        source = backbone_dir(task, cfg) / 'full'
        train = load(source / 'features_train.pt')
        val = load(source / 'features_val.pt')
        validation = {**val, 'embedding': val['masked_embedding']}
        torch.set_float32_matmul_precision('highest')
        print(f'[signature {task}/{variant["name"]}] {len(selection["concept_ids"])} validation concepts, '
              f'{len(schema["concepts"])} items', flush=True)
        artifact, history = fit(train, validation, labels, schema, cfg, device,
                                masked_train=train['masked_embedding'], model_cls=variant['model_cls'])
        retained = {str(item['concept_id']) for item in artifact['spec']['items']}
        if retained != set(selection['concept_ids']):
            raise RuntimeError(f'{task} concept layer IDs {sorted(retained)} differ from the frozen validation list')
        save_torch(root / 'model.pt', artifact)
        pd.DataFrame(history).to_csv(root / 'training.csv', index=False)
        write_json(root / 'concept_definitions.json', artifact['spec']['items'])
        write_json(trained_path, dict(
            version=VERSION, task=task, variant=variant['name'], threshold=artifact['threshold'],
            selection=artifact['selection'], concept_ids=selection['concept_ids'],
            n_concepts=len(selection['concept_ids']), n_items=len(artifact['spec']['items']),
            learned_items=sum(item['learnable'] for item in artifact['spec']['items']),
            width=artifact['spec']['width'], bypass=variant['bypass'], backbone_frozen=True))
    evaluate(task, cfg, device, artifact, variant)
    write_json(done_path, dict(
        version=VERSION, task=task, variant=variant['name'], skipped=False, threshold=artifact['threshold'],
        selection=artifact['selection'], concept_ids=selection['concept_ids'],
        n_concepts=len(selection['concept_ids']), n_items=len(artifact['spec']['items']),
        learned_items=sum(item['learnable'] for item in artifact['spec']['items']),
        width=artifact['spec']['width'], n_evaluated=selection['n_evaluated'], mean=selection['mean'],
        std=selection['std'], sigmas=selection['sigmas'], cutoff=selection['cutoff'], bypass=variant['bypass'],
        input_backbone='full', backbone_frozen=True,
        concept_lambda=cfg['concept_lambda'], concept_tolerance=cfg['concept_tolerance']))
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def compare_variants(task, cfg, device):
    """Paired residual-bypass minus bottleneck comparison on saved probabilities."""
    root = Path(cfg['output_root']) / task
    done = {}
    for variant in VARIANTS:
        marker = root / variant['folder'] / 'DONE.json'
        if not marker.exists():
            return
        done[variant['name']] = read_json(marker)
    if any(item.get('skipped') for item in done.values()):
        return
    rows = []
    cuts = {name: done[name]['threshold'] for name in ('bottleneck', 'residual')}
    for split in ('val', 'test'):
        tables = {}
        for variant in VARIANTS:
            tables[variant['name']] = pd.read_csv(root / variant['folder'] / f'predictions_{split}.csv')
        if not tables['bottleneck']['person_id'].astype(str).equals(tables['residual']['person_id'].astype(str)):
            raise ValueError(f'{task}/{split}: bottleneck and residual predictions are not aligned')
        y = tables['bottleneck']['y'].to_numpy()
        boot = PairedF1(y, cfg['bootstrap_samples'], cfg['seed'], device)
        probabilities = {name: {'full': frame['full_probability'].to_numpy(),
                                'masked': frame['masked_probability'].to_numpy()}
                         for name, frame in tables.items()}
        for inputs in ('full', 'masked'):
            for metric in METRICS:
                effect = boot.effects(
                    probabilities['residual'][inputs], probabilities['bottleneck'][inputs],
                    cuts['residual'], cuts['bottleneck'], METRICS[metric])[0]
                rows.append(dict(
                    task=task, split=split, contrast='within_input', model='signature_residual',
                    inputs=inputs, reference='signature_bottleneck', metric=metric, n=len(y), **effect))
        for metric in METRICS:
            gain = boot.intervals(
                _loss_draws(boot, probabilities['bottleneck']['full'], probabilities['bottleneck']['masked'],
                            cuts['bottleneck'], cuts['bottleneck'], metric)
                - _loss_draws(boot, probabilities['residual']['full'], probabilities['residual']['masked'],
                              cuts['residual'], cuts['residual'], metric))[0]
            rows.append(dict(
                task=task, split=split, contrast='robustness', model='signature_residual', inputs='',
                reference='signature_bottleneck', metric=metric, n=len(y), p_value=np.nan, **gain))
        del boot
    pd.DataFrame(rows).to_csv(root / 'signature_comparison.csv', index=False)
    print(f'[signature {task}] residual bypass compared with the bottleneck', flush=True)


def run_signature(task, cfg, device):
    task_root = Path(cfg['output_root']) / task
    relocate_residual_checkpoint(task_root)
    if all(_finished(task_root / variant['folder']) for variant in VARIANTS):
        skipped = any(read_json(task_root / variant['folder'] / 'DONE.json').get('skipped') for variant in VARIANTS)
        if skipped or (task_root / 'signature_comparison.csv').exists():
            print(f'[reuse] {task}/signature both variants', flush=True)
            return
    restoration_path = task_root / 'concept_diagnostics' / 'restoration.csv'
    if not restoration_path.exists():
        raise FileNotFoundError(
            f'{task} concept diagnostics are required before the selected-concept CBM: {restoration_path}')
    if not (task_root / 'mixed_cbm' / 'model.pt').exists():
        raise FileNotFoundError(f'{task} Mixed CBM checkpoint is required for comparison')
    selection = validation_concepts(pd.read_csv(restoration_path, dtype={'concept_id': str}))
    selection['task'] = task
    for variant in VARIANTS:
        run_variant(task, cfg, device, variant, selection)
    compare_variants(task, cfg, device)


def _label(task):
    return TASK_LABELS.get(task, task)


def _cells(frame, metrics):
    rows = []
    for metric in metrics:
        match = frame[frame.metric == metric]
        rows.append('NA' if match.empty else interval(match.iloc[0].to_dict(), signed=True))
    return rows


def _show(row, metric):
    return f'{row[metric]:.4f} [{row[metric + "_ci_lower"]:.4f}, {row[metric + "_ci_upper"]:.4f}]'


def _publish(performance, comparisons):
    """One row per model with both input conditions and the paired masking loss."""
    rows = []
    for (task, model), group in performance.groupby(['task', 'model'], sort=False):
        row = dict(record='performance', task=task, model=model)
        for item in group.to_dict('records'):
            row['threshold'] = item['threshold']
            row['n'] = item['n']
            inputs = item['inputs']
            for metric in METRICS:
                row[f'{inputs}_{metric}'] = item[metric]
                row[f'{inputs}_{metric}_ci_lower'] = item[f'{metric}_ci_lower']
                row[f'{inputs}_{metric}_ci_upper'] = item[f'{metric}_ci_upper']
        losses = comparisons[(comparisons.contrast == 'input_loss')
                             & (comparisons.task == task) & (comparisons.model == model)]
        for item in losses.to_dict('records'):
            metric = item['metric']
            row[f'{metric}_loss'] = item['value']
            row[f'{metric}_loss_ci_lower'] = item['ci_lower']
            row[f'{metric}_loss_ci_upper'] = item['ci_upper']
            row[f'{metric}_loss_p_value'] = item['p_value']
        rows.append(row)
    published = pd.DataFrame(rows)
    extra = comparisons[comparisons.contrast.isin(('within_input', 'robustness'))].copy()
    if extra.empty:
        return published
    extra.insert(0, 'record', extra['contrast'].to_numpy())
    return pd.concat([published, extra], ignore_index=True)


def _selection_record(root, task):
    for variant in VARIANTS:
        path = root / task / variant['folder'] / 'selected_concepts.json'
        if path.exists():
            return read_json(path)
    return None


def _variant_frames(root, tasks, folder):
    available = [task for task in tasks
                 if (root / task / folder / 'performance.csv').exists()
                 and (root / task / folder / 'comparisons.csv').exists()]
    if not available:
        return available, pd.DataFrame(), pd.DataFrame()
    performance = pd.concat([pd.read_csv(root / task / folder / 'performance.csv') for task in available], ignore_index=True)
    comparisons = pd.concat([pd.read_csv(root / task / folder / 'comparisons.csv') for task in available], ignore_index=True)
    return available, performance, comparisons


def _variant_lines(split, available, performance, comparisons, signature_key, signature_label):
    names = {signature_key: signature_label, **BASELINE_NAMES}
    models = (signature_key, 'mixed_cbm', 'full_clinical')
    split_performance = performance[performance.split == split] if len(performance) else performance
    split_comparisons = comparisons[comparisons.split == split] if len(comparisons) else comparisons
    lines = [f'## {signature_label}', '']
    if split_performance.empty:
        return lines + ['This run has not been evaluated.', ''], split_performance, split_comparisons
    lines += ['Cells show the point estimate and 95% CI. Each model uses its own saved threshold.', '']
    body = []
    for task in available:
        for model in models:
            cells = []
            for inputs in ('full', 'masked'):
                row = split_performance[(split_performance.task == task) & (split_performance.model == model)
                                        & (split_performance.inputs == inputs)]
                if row.empty:
                    cells.append('NA')
                else:
                    item = row.iloc[0].to_dict()
                    cells.append('<br>'.join(f'{metric.upper()}: {_show(item, metric)}'
                                             for metric in ('auroc', 'f1', 'ba', 'brier')))
            body.append([_label(task), names[model], *cells])
    lines += [md(['Task', 'Model', 'Full inputs', 'Masked inputs'], body),
              '### Loss after removing nonbasic clinical inputs', '',
              'AUROC, F1, and BA losses are full-input performance minus masked-input performance. '
              'The Brier loss is the increase, masked minus full. Positive values mean the removal worsened that metric.', '']
    losses = split_comparisons[split_comparisons.contrast == 'input_loss']
    loss_rows = []
    for task in available:
        for model in models:
            loss_rows.append([_label(task), names[model],
                              *_cells(losses[(losses.task == task) & (losses.model == model)],
                                      ('auroc', 'f1', 'ba', 'brier'))])
    lines += [md(['Task', 'Model', 'AUROC loss', 'F1 loss', 'BA loss', 'Brier increase'], loss_rows),
              f'### {signature_label} minus all-concept Mixed CBM', '',
              'Positive AUROC, F1, and BA mean this selected-concept run is higher. '
              'A positive Brier difference means its probability error is higher.', '']
    within = split_comparisons[split_comparisons.contrast == 'within_input']
    within_rows = []
    for task in available:
        for inputs, title in (('full', 'Full inputs'), ('masked', 'Masked inputs')):
            within_rows.append([_label(task), title,
                                *_cells(within[(within.task == task) & (within.inputs == inputs)],
                                        ('auroc', 'f1', 'ba', 'brier'))])
    lines += [md(['Task', 'Inputs', 'ΔAUROC', 'ΔF1', 'ΔBA', 'ΔBrier'], within_rows),
              '### Robustness gain', '',
              'Robustness gain is the comparison model’s masking loss minus this run’s masking loss. '
              'A positive value means this run loses less when nonbasic clinical inputs are removed.', '']
    gains = split_comparisons[split_comparisons.contrast == 'robustness']
    gain_rows = []
    for task in available:
        for reference in ('mixed_cbm', 'full_clinical'):
            gain_rows.append([_label(task), BASELINE_NAMES[reference],
                              *_cells(gains[(gains.task == task) & (gains.reference == reference)],
                                      ('auroc', 'f1', 'ba', 'brier'))])
    lines += [md(['Task', 'Comparison', 'AUROC', 'F1', 'BA', 'Brier'], gain_rows)]
    return lines, split_performance, split_comparisons


def make_signature_report(cfg):
    """Aggregate both selected-concept runs. The concept list stays the validation list."""
    root = Path(cfg['output_root'])
    out = root / 'tables'
    out.mkdir(exist_ok=True)
    tasks = [task for task in cfg['tasks']
             if any(_finished(root / task / variant['folder']) for variant in VARIANTS)]
    if not tasks:
        return
    selections = []
    for task in tasks:
        record = _selection_record(root, task)
        if record['concepts']:
            for concept in record['concepts']:
                selections.append(dict(
                    task=task, n_evaluated=record['n_evaluated'], mean=record['mean'], std=record['std'],
                    cutoff=record['cutoff'], n_selected=len(record['concept_ids']), **concept))
        else:
            selections.append(dict(
                task=task, n_evaluated=record['n_evaluated'], mean=record['mean'], std=record['std'],
                cutoff=record['cutoff'], n_selected=0, concept_id='', concept_name='',
                source_fields='', validation_auroc_gain=np.nan, validation_rank=np.nan))
    concepts = pd.DataFrame(selections)
    concepts.to_csv(out / 'SIGNATURE_CONCEPTS.csv', index=False)
    exports = {'bottleneck': 'SIGNATURE_CBM', 'residual': 'SIGNATURE_CBM_RESIDUAL'}
    for split in ('val', 'test'):
        lines = [f'# Selected-concept CBM — {split}', '',
                 f'Completed tasks: {len(tasks)}/{len(cfg["tasks"])}. Seed {cfg["seed"]}. '
                 f'{cfg["bootstrap_samples"]:,} paired, outcome-stratified bootstrap resamples.', '',
                 'Two selected-concept models are trained from the same frozen validation concept list. '
                 'The bottleneck run routes the outcome through those concepts only. '
                 'The residual-bypass run also gives the outcome MLP the part of the frozen embedding orthogonal to the concept-layer weights. '
                 'μ and σ are the mean and population SD (ddof=0) of the individual validation predicted-restoration gains, '
                 f'and an ID is kept only when its validation gain is strictly above {CUTOFF_LABEL}. '
                 'Test gains are not used to choose the concept list.', '',
                 'Masking removes nonbasic clinical inputs; Basic Clinical features and the other eight modalities remain. '
                 'Each run selects its checkpoint and decision threshold on masked validation inputs.', '',
                 '## Validation concept list', '',
                 'Both runs use this list.', '']
        cutoff_rows = []
        for task in tasks:
            record = _selection_record(root, task)
            cutoff_rows.append([_label(task), record['n_evaluated'], f'{record["mean"]:+.6f}',
                                f'{record["std"]:.6f}', f'{record["cutoff"]:+.6f}', len(record['concept_ids'])])
        lines += [md(['Task', 'Evaluated IDs', 'μ', 'σ', CUTOFF_LABEL, 'Selected IDs'], cutoff_rows)]
        for task in tasks:
            chosen = concepts[(concepts.task == task) & (concepts.n_selected > 0)]
            lines += [f'### {_label(task)}', '']
            if chosen.empty:
                lines += [f'No Concept ID exceeded the validation {CUTOFF_LABEL} cutoff, so neither run was trained.', '']
                continue
            lines += [md(['Validation rank', 'Concept ID', 'Concept', 'Source fields', 'Validation ΔAUROC'],
                         [[int(row.validation_rank), row.concept_id, row.concept_name, row.source_fields,
                           f'{row.validation_auroc_gain:+.6f}'] for row in chosen.itertuples()])]
        for variant in VARIANTS:
            available, performance, comparisons = _variant_frames(root, tasks, variant['folder'])
            section, split_performance, split_comparisons = _variant_lines(
                split, available, performance, comparisons, variant['key'], variant['label'])
            lines += section
            if len(split_performance) and len(split_comparisons):
                _publish(split_performance, split_comparisons).to_csv(
                    out / f'{exports[variant["name"]]}_{split}.csv', index=False)
        compared = [task for task in tasks if (root / task / 'signature_comparison.csv').exists()]
        lines += ['## Residual bypass minus bottleneck', '',
                  'Positive AUROC, F1, and BA mean the residual-bypass run is higher. '
                  'A positive Brier difference means its probability error is higher. '
                  'A positive robustness gain means the residual-bypass run loses less when nonbasic clinical inputs are removed.', '']
        if not compared:
            lines += ['The paired comparison has not been computed.', '']
        else:
            paired = pd.concat([pd.read_csv(root / task / 'signature_comparison.csv') for task in compared], ignore_index=True)
            paired = paired[paired.split == split]
            paired.to_csv(out / f'SIGNATURE_BYPASS_COMPARISON_{split}.csv', index=False)
            within_rows, gain_rows = [], []
            for task in compared:
                for inputs, title in (('full', 'Full inputs'), ('masked', 'Masked inputs')):
                    within_rows.append([_label(task), title,
                                        *_cells(paired[(paired.task == task) & (paired.contrast == 'within_input')
                                                       & (paired.inputs == inputs)],
                                                ('auroc', 'f1', 'ba', 'brier'))])
                gain_rows.append([_label(task), *_cells(paired[(paired.task == task) & (paired.contrast == 'robustness')],
                                                        ('auroc', 'f1', 'ba', 'brier'))])
            lines += [md(['Task', 'Inputs', 'ΔAUROC', 'ΔF1', 'ΔBA', 'ΔBrier'], within_rows),
                      md(['Task', 'AUROC robustness', 'F1 robustness', 'BA robustness', 'Brier robustness'], gain_rows)]
        lines += ['## Exports', '',
                  f'[SIGNATURE_CBM_{split}.csv](SIGNATURE_CBM_{split}.csv) is the bottleneck run. '
                  f'[SIGNATURE_CBM_RESIDUAL_{split}.csv](SIGNATURE_CBM_RESIDUAL_{split}.csv) is the residual-bypass run. '
                  f'[SIGNATURE_BYPASS_COMPARISON_{split}.csv](SIGNATURE_BYPASS_COMPARISON_{split}.csv) compares them. '
                  '[SIGNATURE_CONCEPTS.csv](SIGNATURE_CONCEPTS.csv) is the frozen validation list shared by both runs.', '']
        (out / f'SUMMARY_SIGNATURE_{split}.md').write_text('\n'.join(lines))
    print(f'[signature tables] {len(tasks)}/{len(cfg["tasks"])} tasks: {out}', flush=True)
