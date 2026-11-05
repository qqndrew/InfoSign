"""Selected-concept Mixed CBM.

Concept IDs are frozen from validation predicted-restoration gains before
training. The new concept layer contains only those IDs, and the outcome MLP
reads only that concept vector.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from infosign.common import TASK_LABELS, backbone_dir, read_json, write_json, save_torch
from infosign.report import interval, md
from infosign.stats import PairedF1, auroc_selection
from infosign.train import load
from infosign.cbm.diagnostics import METRICS, interval_fields
from infosign.cbm.model import fit, predict, restore


VERSION = 1


def validation_concepts(restoration):
    """Select Concept IDs from the validation predicted-restoration distribution.

    Test rows are removed before the mean and population SD are computed.
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
    for key, values in auroc_selection(gains).items():
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
        cutoff=float(first.auroc_selection_cutoff),
        concept_ids=[item['concept_id'] for item in concepts],
        concepts=concepts)


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


def evaluate(task, cfg, device, artifact):
    """Score full and masked inputs and the paired loss caused by masking."""
    root = Path(cfg['output_root']) / task
    mixed_artifact = load(root / 'mixed_cbm' / 'model.pt')
    signature = restore(artifact, device)
    mixed = restore(mixed_artifact, device)
    full_cut = read_json(backbone_dir(task, cfg) / 'full' / 'TRAINING_DONE.json')['threshold']
    cuts = {'signature_cbm': artifact['threshold'], 'mixed_cbm': mixed_artifact['threshold'],
            'full_clinical': full_cut}
    performance, comparisons = [], []
    for split in ('val', 'test'):
        cache = load(backbone_dir(task, cfg) / 'full' / f'features_{split}.pt')
        y = cache['y'].numpy()
        boot = PairedF1(y, cfg['bootstrap_samples'], cfg['seed'], device)
        probabilities = {
            'signature_cbm': {
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
                    probabilities['signature_cbm'][inputs], probabilities['mixed_cbm'][inputs],
                    cuts['signature_cbm'], cuts['mixed_cbm'], METRICS[metric])[0]
                comparisons.append(dict(
                    task=task, split=split, contrast='within_input', model='signature_cbm',
                    inputs=inputs, reference='mixed_cbm', metric=metric, n=len(y), **effect))
        for reference in ('mixed_cbm', 'full_clinical'):
            for metric in METRICS:
                gain = boot.intervals(
                    _loss_draws(boot, probabilities[reference]['full'], probabilities[reference]['masked'],
                                cuts[reference], cuts[reference], metric)
                    - _loss_draws(boot, probabilities['signature_cbm']['full'],
                                  probabilities['signature_cbm']['masked'],
                                  cuts['signature_cbm'], cuts['signature_cbm'], metric))[0]
                comparisons.append(dict(
                    task=task, split=split, contrast='robustness', model='signature_cbm', inputs='',
                    reference=reference, metric=metric, n=len(y), p_value=np.nan, **gain))
        del boot
        print(f'[signature {task}/{split}] full and masked inputs scored', flush=True)
    folder = root / 'signature_cbm'
    pd.DataFrame(performance).to_csv(folder / 'performance.csv', index=False)
    pd.DataFrame(comparisons).to_csv(folder / 'comparisons.csv', index=False)


def run_signature(task, cfg, device):
    root = Path(cfg['output_root']) / task / 'signature_cbm'
    root.mkdir(parents=True, exist_ok=True)
    done_path = root / 'DONE.json'
    if done_path.exists() and read_json(done_path).get('version') == VERSION:
        required = () if read_json(done_path).get('skipped') else ('model.pt', 'performance.csv', 'comparisons.csv')
        if all((root / name).exists() for name in required):
            print(f'[reuse] {task}/signature_cbm', flush=True)
            return
    restoration_path = Path(cfg['output_root']) / task / 'concept_diagnostics' / 'restoration.csv'
    if not restoration_path.exists():
        raise FileNotFoundError(
            f'{task} concept diagnostics are required before the selected-concept CBM: {restoration_path}')
    if not (Path(cfg['output_root']) / task / 'mixed_cbm' / 'model.pt').exists():
        raise FileNotFoundError(f'{task} Mixed CBM checkpoint is required for comparison')
    restoration = pd.read_csv(restoration_path, dtype={'concept_id': str})
    selection = validation_concepts(restoration)
    selection['task'] = task
    write_json(root / 'selected_concepts.json', selection)
    if not selection['concept_ids']:
        write_json(done_path, dict(
            version=VERSION, task=task, skipped=True, reason='no_validation_concept_above_cutoff',
            n_evaluated=selection['n_evaluated'], mean=selection['mean'], std=selection['std'],
            cutoff=selection['cutoff']))
        print(f'[signature {task}] no validation concept exceeded μ + 2σ; skipped', flush=True)
        return
    trained_path = root / 'TRAINING_DONE.json'
    trained = read_json(trained_path) if trained_path.exists() else {}
    reuse_training = (trained.get('version') == VERSION and (root / 'model.pt').exists()
                      and trained.get('concept_ids') == selection['concept_ids'])
    if reuse_training:
        artifact = load(root / 'model.pt')
        print(f'[reuse training] {task}/signature_cbm', flush=True)
    else:
        labels = pd.read_csv(Path(cfg['output_root']) / 'labels' / task / 'labels.csv', dtype=str, keep_default_na=False)
        schema = restrict_schema(read_json(Path(cfg['output_root']) / 'labels' / task / 'schema.json'), selection['concept_ids'])
        source = backbone_dir(task, cfg) / 'full'
        train = load(source / 'features_train.pt')
        val = load(source / 'features_val.pt')
        validation = {**val, 'embedding': val['masked_embedding']}
        torch.set_float32_matmul_precision('highest')
        print(f'[signature {task}] {len(selection["concept_ids"])} validation concepts, {len(schema["concepts"])} items', flush=True)
        artifact, history = fit(train, validation, labels, schema, cfg, device, masked_train=train['masked_embedding'])
        retained = {str(item['concept_id']) for item in artifact['spec']['items']}
        if retained != set(selection['concept_ids']):
            raise RuntimeError(f'{task} concept layer IDs {sorted(retained)} differ from the frozen validation list')
        save_torch(root / 'model.pt', artifact)
        pd.DataFrame(history).to_csv(root / 'training.csv', index=False)
        write_json(root / 'concept_definitions.json', artifact['spec']['items'])
        write_json(trained_path, dict(
            version=VERSION, task=task, threshold=artifact['threshold'], selection=artifact['selection'],
            concept_ids=selection['concept_ids'], n_concepts=len(selection['concept_ids']),
            n_items=len(artifact['spec']['items']),
            learned_items=sum(item['learnable'] for item in artifact['spec']['items']),
            width=artifact['spec']['width'], bypass=False, backbone_frozen=True))
    evaluate(task, cfg, device, artifact)
    write_json(done_path, dict(
        version=VERSION, task=task, skipped=False, threshold=artifact['threshold'],
        selection=artifact['selection'], concept_ids=selection['concept_ids'],
        n_concepts=len(selection['concept_ids']),
        n_items=len(artifact['spec']['items']),
        learned_items=sum(item['learnable'] for item in artifact['spec']['items']),
        width=artifact['spec']['width'],
        n_evaluated=selection['n_evaluated'], mean=selection['mean'], std=selection['std'],
        cutoff=selection['cutoff'], bypass=False, input_backbone='full', backbone_frozen=True,
        concept_lambda=cfg['concept_lambda'], concept_tolerance=cfg['concept_tolerance']))


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


def make_signature_report(cfg):
    """Aggregate completed selected-concept CBMs. The concept list stays the validation list."""
    root = Path(cfg['output_root'])
    out = root / 'tables'
    out.mkdir(exist_ok=True)
    tasks = []
    for task in cfg['tasks']:
        marker = root / task / 'signature_cbm' / 'DONE.json'
        if marker.exists() and read_json(marker).get('version') == VERSION:
            tasks.append(task)
    if not tasks:
        return
    selections = []
    for task in tasks:
        record = read_json(root / task / 'signature_cbm' / 'selected_concepts.json')
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
    available = [task for task in tasks
                 if (root / task / 'signature_cbm' / 'performance.csv').exists()
                 and (root / task / 'signature_cbm' / 'comparisons.csv').exists()]
    performance = (pd.concat([pd.read_csv(root / task / 'signature_cbm' / 'performance.csv') for task in available],
                             ignore_index=True) if available else pd.DataFrame())
    comparisons = (pd.concat([pd.read_csv(root / task / 'signature_cbm' / 'comparisons.csv') for task in available],
                             ignore_index=True) if available else pd.DataFrame())
    names = {'signature_cbm': 'Selected-concept CBM', 'mixed_cbm': 'All-concept Mixed CBM',
             'full_clinical': 'Full Clinical model'}
    for split in ('val', 'test'):
        split_performance = performance[performance.split == split] if len(performance) else performance
        split_comparisons = comparisons[comparisons.split == split] if len(comparisons) else comparisons
        if len(split_performance) and len(split_comparisons):
            _publish(split_performance, split_comparisons).to_csv(out / f'SIGNATURE_CBM_{split}.csv', index=False)
        lines = [f'# Selected-concept CBM — {split}', '',
                 f'Completed tasks: {len(tasks)}/{len(cfg["tasks"])}. Seed {cfg["seed"]}. '
                 f'{cfg["bootstrap_samples"]:,} paired, outcome-stratified bootstrap resamples.', '',
                 'Concept IDs are frozen from validation predicted-restoration ΔAUROC before training. '
                 'μ and σ are the mean and population SD (ddof=0) of the individual validation gains, '
                 'and an ID is kept only when its validation gain is strictly above μ + 2σ. '
                 'The same IDs define the concept layer for training, validation, and test. '
                 'Test gains are not used to compute μ, σ, the cutoff, or the selected IDs.', '',
                 'The selected-concept model is a new Mixed CBM on the frozen Full Clinical backbone. '
                 'Its concept layer and outcome MLP contain only the retained concepts. '
                 'There is no bypass from the fused embedding to the outcome. '
                 'Masking removes nonbasic clinical inputs; Basic Clinical features and the other eight modalities remain. '
                 'The checkpoint and decision threshold are selected on masked validation inputs.', '',
                 '## Validation concept list', '',
                 'This list is identical for the validation and test reports.', '']
        cutoff_rows = []
        for task in tasks:
            record = read_json(root / task / 'signature_cbm' / 'selected_concepts.json')
            cutoff_rows.append([_label(task), record['n_evaluated'], f'{record["mean"]:+.6f}',
                                f'{record["std"]:.6f}', f'{record["cutoff"]:+.6f}', len(record['concept_ids'])])
        lines += [md(['Task', 'Evaluated IDs', 'μ', 'σ', 'μ + 2σ', 'Selected IDs'], cutoff_rows)]
        for task in tasks:
            chosen = concepts[(concepts.task == task) & (concepts.n_selected > 0)]
            lines += [f'### {_label(task)}', '']
            if chosen.empty:
                lines += ['No Concept ID exceeded the validation μ + 2σ cutoff, so no selected-concept CBM was trained.', '']
                continue
            lines += [md(['Validation rank', 'Concept ID', 'Concept', 'Source fields', 'Validation ΔAUROC'],
                         [[int(row.validation_rank), row.concept_id, row.concept_name, row.source_fields,
                           f'{row.validation_auroc_gain:+.6f}'] for row in chosen.itertuples()])]
        if split_performance.empty:
            lines += ['No trained selected-concept CBM has been evaluated.', '']
        else:
            lines += ['## Performance', '',
                      'Cells show the point estimate and 95% CI. Each model uses its own saved threshold.', '']
            header = ['Task', 'Model', 'Full inputs', 'Masked inputs']
            body = []
            for task in available:
                for model in ('signature_cbm', 'mixed_cbm', 'full_clinical'):
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
            lines += [md(header, body),
                      '## Loss after removing nonbasic clinical inputs', '',
                      'AUROC, F1, and BA losses are full-input performance minus masked-input performance. '
                      'The Brier loss is the increase, masked minus full. Positive values mean the removal worsened that metric.', '']
            loss_rows = []
            losses = split_comparisons[split_comparisons.contrast == 'input_loss']
            for task in available:
                for model in ('signature_cbm', 'mixed_cbm', 'full_clinical'):
                    loss_rows.append([_label(task), names[model],
                                      *_cells(losses[(losses.task == task) & (losses.model == model)],
                                              ('auroc', 'f1', 'ba', 'brier'))])
            lines += [md(['Task', 'Model', 'AUROC loss', 'F1 loss', 'BA loss', 'Brier increase'], loss_rows),
                      '## Selected-concept CBM minus all-concept Mixed CBM', '',
                      'Positive AUROC, F1, and BA mean the selected-concept CBM is higher. '
                      'A positive Brier difference means its probability error is higher.', '']
            within_rows = []
            within = split_comparisons[split_comparisons.contrast == 'within_input']
            for task in available:
                for inputs, title in (('full', 'Full inputs'), ('masked', 'Masked inputs')):
                    within_rows.append([_label(task), title,
                                        *_cells(within[(within.task == task) & (within.inputs == inputs)],
                                                ('auroc', 'f1', 'ba', 'brier'))])
            lines += [md(['Task', 'Inputs', 'ΔAUROC', 'ΔF1', 'ΔBA', 'ΔBrier'], within_rows),
                      '## Robustness gain', '',
                      'Robustness gain is the comparison model’s masking loss minus the selected-concept CBM’s masking loss. '
                      'A positive value means the selected-concept CBM loses less when nonbasic clinical inputs are removed. '
                      'Intervals condition on the fitted models and the validation-chosen concept list.', '']
            gain_rows = []
            gains = split_comparisons[split_comparisons.contrast == 'robustness']
            for task in available:
                for reference in ('mixed_cbm', 'full_clinical'):
                    gain_rows.append([_label(task), names[reference],
                                      *_cells(gains[(gains.task == task) & (gains.reference == reference)],
                                              ('auroc', 'f1', 'ba', 'brier'))])
            lines += [md(['Task', 'Comparison', 'AUROC', 'F1', 'BA', 'Brier'], gain_rows)]
        lines += ['## Exports', '',
                  f'[SIGNATURE_CBM_{split}.csv](SIGNATURE_CBM_{split}.csv) contains one performance value per input '
                  'and the paired masking losses. [SIGNATURE_CONCEPTS.csv](SIGNATURE_CONCEPTS.csv) is the frozen validation list.', '']
        (out / f'SUMMARY_SIGNATURE_{split}.md').write_text('\n'.join(lines))
    print(f'[signature tables] {len(tasks)}/{len(cfg["tasks"])} tasks: {out}', flush=True)
