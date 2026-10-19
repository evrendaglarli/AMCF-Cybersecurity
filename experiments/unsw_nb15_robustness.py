"""Repeated, leakage-aware, real UNSW-NB15 pilot robustness evaluation.

This evaluates an AMCF-lite *fusion proxy*, NOT the AMCF manuscript architecture.
Validation-thresholded flow-level operating points; no time-to-containment,
SOC FP/hour, attack-chain memory, or automated response is implemented.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from unsw_nb15_feasibility import (
    load_release, subsample, preprocessing, clean_features,
    fixed_fpr_threshold, metrics, sha256,
)


def fusion(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Equal-weight logit score fusion; clipped to avoid infinities."""
    eps = 1e-6
    left = np.clip(np.asarray(left, dtype=float), eps, 1-eps)
    right = np.clip(np.asarray(right, dtype=float), eps, 1-eps)
    z = (np.log(left / (1-left)) + np.log(right / (1-right))) / 2
    return 1 / (1 + np.exp(-z))


def wilson_ci(numerator: int, denominator: int, z: float = 1.959963984540054):
    """Two-sided Wilson score confidence interval for an observed proportion."""
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    p = numerator / denominator
    a = z * z / denominator
    center = (p + a/2) / (1+a)
    half = z * np.sqrt((p*(1-p) + z*z/(4*denominator))/denominator) / (1+a)
    return (float(max(0., center-half)), float(min(1., center+half)))


def paired_delta_bootstrap(
    y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
    *, class_label: int, repetitions: int = 3000, seed: int = 2026,
):
    """Stratified paired nonparametric CI for recall or FPR difference (A - B).

    Multinomial bootstrap of the observed 2x2 paired prediction table within
    the requested TRUE class: exact in distribution to resampling paired rows.
    No independence is assumed between models' predictions on the same event.
    """
    sub = np.asarray(y) == class_label
    a, b = np.asarray(pred_a)[sub], np.asarray(pred_b)[sub]
    if len(a) == 0:
        raise ValueError("No examples in requested class")
    counts = np.bincount(a.astype(int)*2 + b.astype(int), minlength=4)
    rng = np.random.default_rng(seed)
    sampled = rng.multinomial(len(a), counts / len(a), size=repetitions)
    # Categories 10 (+1) and 01 (-1) contribute to difference.
    differences = (sampled[:, 2] - sampled[:, 1]) / len(a)
    return float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))


def one_seed(train_full, test_full, features, *, seed: int, train_limit: int,
             budgets: list[float], c_triage: float, c_fn: float, boot: int):
    train = subsample(train_full, train_limit, seed)
    # Fixed complete public test partition for all seeds, not independent tests.
    test = test_full
    development, tune = train_test_split(train, test_size=0.15, random_state=seed,
                                         stratify=train['label'])
    fit, calibration = train_test_split(development, test_size=0.15/0.85,
                                        random_state=seed+1,
                                        stratify=development['label'])
    transformer, nums, cats = preprocessing(fit, features)
    x_fit = transformer.fit_transform(clean_features(fit, nums, cats))
    x_cal = transformer.transform(clean_features(calibration, nums, cats))
    x_tune = transformer.transform(clean_features(tune, nums, cats))
    x_test = transformer.transform(clean_features(test, nums, cats))
    y_fit = fit.label.to_numpy(dtype=int)
    y_cal = calibration.label.to_numpy(dtype=int)
    y_tune = tune.label.to_numpy(dtype=int)
    y_test = test.label.to_numpy(dtype=int)

    models = {
        'logistic_baseline': LogisticRegression(max_iter=500, random_state=seed),
        'hgb_baseline': HistGradientBoostingClassifier(
            max_iter=100, max_leaf_nodes=15, random_state=seed),
    }
    scores = {}
    for name, model in models.items():
        model.fit(x_fit, y_fit)
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(model.predict_proba(x_cal)[:, 1], y_cal)
        scores[name] = (
            np.asarray(iso.predict(model.predict_proba(x_tune)[:, 1])),
            np.asarray(iso.predict(model.predict_proba(x_test)[:, 1])),
        )
    scores['amcf_lite_fusion'] = (
        fusion(scores['logistic_baseline'][0], scores['hgb_baseline'][0]),
        fusion(scores['logistic_baseline'][1], scores['hgb_baseline'][1]),
    )
    rows, differences = [], []
    for budget in budgets:
        chosen_preds = {}
        for name, (ptune, ptest) in scores.items():
            validation_threshold = fixed_fpr_threshold(ptune, y_tune, budget)
            threshold = validation_threshold
            if name == 'amcf_lite_fusion':
                threshold = max(threshold, c_triage / c_fn)
            tune_false_pos = ((ptune >= threshold) & (y_tune == 0)).sum()
            n_tune_benign = (y_tune == 0).sum()
            reported = metrics(y_test, ptest, threshold)
            pos_ci = wilson_ci(reported['tp'], reported['tp']+reported['fn'])
            neg_ci = wilson_ci(reported['fp'], reported['fp']+reported['tn'])
            rows.append({
                'seed': seed, 'target_validation_fpr': budget, 'model': name,
                'train_rows': len(train), 'fit_rows': len(fit),
                'calibration_rows': len(calibration), 'tuning_rows': len(tune),
                'test_rows': len(test),
                'validation_realized_fpr': float(tune_false_pos / n_tune_benign),
                'recall_ci_low': pos_ci[0], 'recall_ci_high': pos_ci[1],
                'fpr_ci_low': neg_ci[0], 'fpr_ci_high': neg_ci[1],
                **reported,
            })
            chosen_preds[name] = ptest >= threshold
        for comparator in ('logistic_baseline', 'hgb_baseline'):
            proxy = chosen_preds['amcf_lite_fusion']
            base = chosen_preds[comparator]
            pos = y_test == 1
            neg = ~pos
            recall_delta_ci = paired_delta_bootstrap(
                y_test, proxy, base, class_label=1, repetitions=boot, seed=seed+17)
            fpr_delta_ci = paired_delta_bootstrap(
                y_test, proxy, base, class_label=0, repetitions=boot, seed=seed+31)
            differences.append({
                'seed':seed, 'target_validation_fpr': budget,
                'comparison':f'amcf_lite_minus_{comparator}',
                'delta_recall':float(np.mean(proxy[pos])-np.mean(base[pos])),
                'delta_fpr':float(np.mean(proxy[neg])-np.mean(base[neg])),
                'delta_recall_ci_low': recall_delta_ci[0],
                'delta_recall_ci_high': recall_delta_ci[1],
                'delta_fpr_ci_low': fpr_delta_ci[0],
                'delta_fpr_ci_high': fpr_delta_ci[1],
            })
    return rows, differences, {'seed': seed, 'fit':len(fit), 'cal':len(calibration),
                                'tune':len(tune), 'encoded_feature_count':x_fit.shape[1]}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',type=Path,required=True)
    parser.add_argument('--test',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=Path('results/feasibility/robustness'))
    parser.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46])
    parser.add_argument('--max-train',type=int,default=20000,
                        help='Training subset per seed, 0 uses full official train')
    parser.add_argument('--budgets',nargs='+',type=float,default=[0.01,0.05,0.10],
                        help='Validation-class FPR targets (not observed test rates)')
    parser.add_argument('--bootstrap',type=int,default=3000)
    parser.add_argument('--c-triage',type=float,default=3.0)
    parser.add_argument('--c-fn',type=float,default=15.0)
    args=parser.parse_args(argv)
    if args.max_train not in [0] and args.max_train < 200:
        parser.error('--max-train must be 0 or >=200')
    if len(set(args.seeds))!=len(args.seeds):
        parser.error('--seeds must be distinct')
    if not all(0<v<1 for v in args.budgets):
        parser.error('FPR budgets must lie strictly between 0 and 1')
    if args.bootstrap<100:
        parser.error('--bootstrap must be >=100')
    if args.c_triage<=0 or args.c_fn<=0:
        parser.error('costs must be strictly positive')
    return args


def main(argv=None):
    args=parse_args(argv)
    train, test, feats=load_release(args.train,args.test)
    all_rows=[]; all_diffs=[]; splits=[]
    for i, seed in enumerate(args.seeds, 1):
        print(f'[{i}/{len(args.seeds)}] Training/calibrating seed {seed}',flush=True)
        rows,diffs,split = one_seed(train,test,feats,seed=seed,train_limit=args.max_train,
                                    budgets=args.budgets,c_triage=args.c_triage,
                                    c_fn=args.c_fn,boot=args.bootstrap)
        all_rows.extend(rows);all_diffs.extend(diffs);splits.append(split)
    args.out.mkdir(parents=True,exist_ok=True)
    details=pd.DataFrame(all_rows); comparisons=pd.DataFrame(all_diffs)
    details.to_csv(args.out/'seed_operating_points.csv',index=False)
    comparisons.to_csv(args.out/'paired_differences.csv',index=False)
    aggregate=(details.groupby(['target_validation_fpr','model'],as_index=False)
               .agg(seeds=('seed','count'),recall_mean=('recall','mean'),
                    recall_min=('recall','min'),recall_max=('recall','max'),
                    test_fpr_mean=('fpr','mean'),test_fpr_min=('fpr','min'),
                    test_fpr_max=('fpr','max'),auc_mean=('auc_roc','mean'),
                    ap_mean=('average_precision','mean'),
                    tuning_fpr_mean=('validation_realized_fpr','mean')))
    aggregate.to_csv(args.out/'seed_summary.csv',index=False)
    provenance={
        'scope':'AMCF-lite real UNSW-NB15 technical robustness; NOT original R20 experiments',
        'train_sha256':sha256(args.train),'test_sha256':sha256(args.test),
        'input_train_rows':len(train),'input_test_rows':len(test),
        'seeds':args.seeds,'max_train':args.max_train,'budgets':args.budgets,
        'bootstrap_resamples':args.bootstrap,'cost_ratio':args.c_triage/args.c_fn,
        'splits':splits,'python':platform.python_version(),
        'numpy':np.__version__,'pandas':pd.__version__,
        'scikit_learn':sklearn.__version__,'scipy':scipy.__version__,
        'statistical_method':(
            'Wilson 95% CI for held-out test flow recall/FPR; '
            'paired class-stratified multinomial nonparametric bootstrap for '
            'AMCF-lite minus baseline differences. '
            'All seeds reuse the same fixed public test partition; '
            'seed repetitions are not independent new holdout datasets.'),
        'caveat':(
            'Validation FPR targets do not imply held-out test FPR control; '
            'never choose a threshold using the test labels. '
            'No temporal/incident, drift, safety, human-factors, or full AMCF evaluation.'),
    }
    (args.out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(aggregate.round(5).to_string(index=False))
    print('Results:',args.out,flush=True)

if __name__=='__main__':
    main()