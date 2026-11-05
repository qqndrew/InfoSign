"""Classification metrics, train-only balancing, and paired participant inference."""
import numpy as np
import torch
from torch.utils.data import Sampler


def f1(y, probability, cut):
    y = np.asarray(y, bool)
    pred = np.asarray(probability) >= cut
    return 2 * (pred & y).sum(axis=-1) / np.maximum(1, pred.sum(axis=-1) + y.sum())


def auroc(y, probability):
    """Tie-aware AUROC for one or several probability vectors."""
    y = np.asarray(y, int); p = np.asarray(probability)
    rows = np.atleast_2d(p)
    values = np.full(len(rows), np.nan)
    denominator = int((y == 1).sum()) * int((y == 0).sum())
    if denominator:
        for j, row in enumerate(rows):
            order = np.argsort(row, kind='stable'); truth = y[order]
            starts = np.r_[0, np.flatnonzero(np.diff(row[order])) + 1]
            ends = np.r_[starts[1:], len(y)]
            negative = np.r_[0, (truth == 0).cumsum()]
            credit = np.repeat((negative[starts] + negative[ends]) / 2, ends-starts)
            values[j] = credit[truth == 1].sum() / denominator
    return float(values[0]) if p.ndim == 1 else values


def auroc_selection(gains, eligible=None, sigmas=2):
    """Select individual IDs strictly above their own distribution's μ + kσ.

    Call once per task, split and intervention population, without a joint/all-ID
    row. σ is the population SD (ddof=0) and k defaults to 2. This is descriptive
    selection, not a significance test; zero/negative effects are not additionally
    filtered.
    """
    gains = np.asarray(gains, float)
    valid = np.isfinite(gains)
    if eligible is not None: valid &= np.asarray(eligible, bool)
    n = int(valid.sum()); values = gains[valid]
    mean = float(values.mean()) if n else np.nan
    std = float(values.std(ddof=0)) if n else np.nan
    cutoff = mean + sigmas*std
    rank = np.full(len(gains), np.nan)
    rank[valid] = np.searchsorted(np.sort(-values), -values, side='left') + 1
    return dict(auroc_rank_within_split=rank,
                auroc_selection_n=np.full(len(gains), n),
                auroc_selection_mean=np.full(len(gains), mean),
                auroc_selection_std=np.full(len(gains), std),
                auroc_selection_cutoff=np.full(len(gains), cutoff),
                auroc_selected=valid & (gains > cutoff))


def threshold(y, probability):
    """Maximize validation F1; use the highest threshold when scores tie."""
    p = np.asarray(probability, dtype=float); y = np.asarray(y, dtype=int)
    order = np.argsort(-p, kind='stable'); p = p[order]; y = y[order]
    ends = np.flatnonzero(np.r_[p[1:] != p[:-1], True])
    scores = 2 * y.cumsum()[ends] / np.maximum(1, ends + 1 + y.sum())
    return float(p[ends[np.argmax(scores)]])


def balanced_indices(y, seed):
    """Keep every training participant, duplicate the smaller class to equality."""
    y = np.asarray(y); rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == k) for k in (0, 1)]
    if any(len(g) == 0 for g in groups):
        raise ValueError('Training needs both outcome classes for oversampling')
    size = max(map(len, groups))
    indices = np.concatenate([np.r_[g, rng.choice(g, size - len(g), replace=True)] for g in groups])
    return rng.permutation(indices)


class BalancedSampler(Sampler):
    def __init__(self, y, seed):
        self.y = np.asarray(y); self.seed = seed; self.epoch = 0
    def __len__(self):
        return 2 * max(int((self.y == k).sum()) for k in (0, 1))
    def __iter__(self):
        return iter(balanced_indices(self.y, self.seed + self.epoch).tolist())


def bh_adjust(p):
    p = np.asarray(p, float); order = np.argsort(p); n = len(p)
    adjusted = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    result = np.empty(n); result[order] = np.clip(adjusted, 0, 1)
    return result


class PairedF1:
    """Shared stratified bootstrap draws; paired prediction-swap permutation p.

    Evaluation participants occur once in the input. Resampling here estimates
    uncertainty, and never changes the evaluated prevalence or trains a model.
    Each swap exchanges two predictions for one participant (probabilities for
    Brier, thresholded decisions for F1/BA). It tests paired
    prediction exchangeability conditional on the fitted models/thresholds.
    """
    def __init__(self, y, samples=10000, seed=42, device='cpu', allow_single_class=False):
        self.y = np.asarray(y, int); self.device = device
        self.samples = samples; rng = np.random.default_rng(seed)
        weights = np.ones((samples + 1, len(y)), np.float64)
        for k in (0, 1):
            idx = np.flatnonzero(self.y == k)
            if not len(idx):
                if allow_single_class: continue
                raise ValueError('Evaluation needs both outcome classes')
            weights[1:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1/len(idx)), samples)
        self.w = torch.as_tensor(weights, device=device)
        self.truth = torch.as_tensor(self.y, dtype=torch.float64, device=device)
        self.positive = self.w @ self.truth
        self.swaps = torch.as_tensor(rng.integers(0, 2, (samples, len(y))),
                                     dtype=torch.float64, device=device)

    def decisions(self, p, cut):
        return torch.as_tensor(np.atleast_2d(np.asarray(p) >= cut), dtype=torch.float64, device=self.device)

    @staticmethod
    def score(tp, predicted, positive, total, metric):
        if metric == 'balanced_accuracy':
            return .5*(tp/positive + (total-positive-predicted+tp)/(total-positive))
        return 2*tp/(predicted+positive).clamp_min(1)

    def draws(self, p, cut, metric='f1'):
        if metric in ('balanced_accuracy', 'auroc') and len(np.unique(self.y))<2:
            return np.full((len(np.atleast_2d(p)),self.samples+1),np.nan)
        if metric == 'auroc':
            return self.auroc_draws(p)
        if metric == 'brier':
            p = torch.as_tensor(np.atleast_2d(p), dtype=torch.float64, device=self.device)
            return ((p-self.truth).square() @ self.w.T / self.w.sum(1)).cpu().numpy()
        d = self.decisions(p, cut)
        tp = (d * self.truth) @ self.w.T
        predicted = d @ self.w.T
        return self.score(tp, predicted, self.positive, self.w.sum(1), metric).cpu().numpy()

    def auroc_draws(self, p):
        """Weighted rank AUROC with the same stratified draws as F1/BA.

        Scores are sorted once per intervention, not once per bootstrap sample.
        Tied positive-negative pairs receive half credit. Small batches bound
        temporary tensors while using the selected CPU/GPU device.
        """
        probabilities = torch.as_tensor(np.atleast_2d(p), dtype=torch.float64, device=self.device)
        count, n = probabilities.shape
        budget = 16_000_000 if self.w.is_cuda else 2_000_000
        batch = max(1, min(count, budget // ((self.samples+1)*n)))
        position = torch.arange(n, device=self.device)
        denominator = self.positive * (self.w.sum(1)-self.positive)
        results = []
        for start in range(0, count, batch):
            scores, order = probabilities[start:start+batch].sort(dim=1, stable=True)
            truth = self.truth[order]
            same = scores[:,1:] == scores[:,:-1]
            first = torch.cat([torch.ones_like(same[:,:1]), ~same], dim=1)
            last = torch.cat([~same, torch.ones_like(same[:,:1])], dim=1)
            starts = torch.where(first, position, 0).cummax(1).values
            ends = torch.where(last, position, n-1).flip(1).cummin(1).values.flip(1)
            weights = self.w[None].expand(len(scores),-1,-1).gather(2,order[:,None].expand(-1,len(self.w),-1))
            negative = (weights*(1-truth[:,None])).cumsum(2)
            cumulative = torch.cat([torch.zeros_like(negative[:,:,:1]),negative],dim=2)
            low = cumulative.gather(2,starts[:,None].expand(-1,len(self.w),-1))
            high = cumulative.gather(2,(ends+1)[:,None].expand(-1,len(self.w),-1))
            result = (weights*truth[:,None]*(low+high)*.5).sum(2)/denominator
            results.append(result.cpu().numpy())
        return np.concatenate(results)

    def pvalues(self, a, b, cut_a, cut_b, metric='f1'):
        if metric == 'auroc':
            # AUROC is reported with paired bootstrap CIs, not permutation p/q.
            return np.full(max(len(np.atleast_2d(a)),len(np.atleast_2d(b))),np.nan)
        if metric == 'balanced_accuracy' and len(np.unique(self.y))<2:
            return np.full(max(len(np.atleast_2d(a)),len(np.atleast_2d(b))),np.nan)
        if metric == 'brier':
            a = torch.as_tensor(np.atleast_2d(a), dtype=torch.float64, device=self.device)
            b = torch.as_tensor(np.atleast_2d(b), dtype=torch.float64, device=self.device)
            difference = (a-self.truth).square() - (b-self.truth).square()
            observed = difference.mean(1, keepdim=True)
            null = (difference.sum(1, keepdim=True)-2*difference @ self.swaps.T)/len(self.y)
            return ((1+(null.abs() >= observed.abs()-1e-12).sum(1))/(self.samples+1)).cpu().numpy()
        a = self.decisions(a, cut_a); b = self.decisions(b, cut_b)
        a, b = torch.broadcast_tensors(a, b)
        # Swapping only changes predicted positives and true positives.
        change = (b-a) @ self.swaps.T
        change_tp = ((b-a)*self.truth) @ self.swaps.T
        ap, bp = a.sum(1, keepdim=True), b.sum(1, keepdim=True)
        at, bt = (a*self.truth).sum(1, keepdim=True), (b*self.truth).sum(1, keepdim=True)
        npos = self.truth.sum()
        def score(tp, predicted): return self.score(tp, predicted, npos, len(self.y), metric)
        observed = score(at, ap) - score(bt, bp)
        null = score(at+change_tp, ap+change) - score(bt-change_tp, bp-change)
        return ((1+(null.abs() >= observed.abs()-1e-12).sum(1))/(self.samples+1)).cpu().numpy()

    @staticmethod
    def intervals(draws):
        draws = np.atleast_2d(draws)
        ci = np.quantile(draws[:, 1:], [.025, .975], axis=1)
        return [dict(value=float(d[0]), ci_lower=float(lo), ci_upper=float(hi))
                for d, lo, hi in zip(draws, *ci)]

    def performance(self, p, cut, metric='f1'):
        return self.intervals(self.draws(p, cut, metric))[0]

    def effects(self, a, b, cut_a, cut_b, metric='f1'):
        """Effect = metric(a) - metric(b); all draws use the same participants."""
        rows = self.intervals(self.draws(a, cut_a, metric)-self.draws(b, cut_b, metric))
        for row, p in zip(rows, self.pvalues(a, b, cut_a, cut_b, metric)):
            row['p_value'] = float(p)
        return rows
