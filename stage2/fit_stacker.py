"""Post-hoc account stacker for separate-mode runs: gradient boosting on the base account score plus
per-account aggregates of transaction risk. Fit on each seed's validation-window export (scores_val/),
evaluated on its test-window export (scores/). Inference must aggregate transaction risk over the same window."""
import argparse,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
import config
from stage2.evaluate import compute_metrics,find_optimal_threshold,aggregate_seed_metrics

AGGREGATES=('mean','max','top3','log_count','n50','n90')

def side(tx,col,tag):
    risk=tx.groupby(col)['transaction_risk_score']
    top=tx.sort_values('transaction_risk_score',ascending=False).groupby(col).head(3)
    out=pd.DataFrame({'mean':risk.mean(),'max':risk.max(),'top3':top.groupby(col)['transaction_risk_score'].mean(),
                      'log_count':np.log1p(risk.size()),
                      'n50':(tx.transaction_risk_score>.5).groupby(tx[col]).sum(),
                      'n90':(tx.transaction_risk_score>.9).groupby(tx[col]).sum()})
    return out.add_prefix(tag+'_')

def load(scores_dir,target,labels,cutoff):
    acc=pd.read_parquet(scores_dir/'account_scores.parquet')
    acc=acc[acc[f'{target}_account_eligible']].merge(labels,on='account_id')
    tx=pd.read_parquet(scores_dir/'transaction_scores.parquet',
                       columns=['sender_account_id','receiver_account_id','timestamp_unix','transaction_risk_score'])
    feats=side(tx,'sender_account_id','out').join(side(tx,'receiver_account_id','in'),how='outer')
    acc=acc.merge(feats,left_on='account_id',right_index=True,how='left').fillna(0)
    late=tx[tx.timestamp_unix>=cutoff]
    acc['late']=acc.account_id.isin(set(late.sender_account_id)|set(late.receiver_account_id))
    p=acc.base_account_risk_score.clip(1e-6,1-1e-6)
    acc['base_logit']=np.log(p/(1-p))
    return acc

def columns():
    return ['base_logit']+[f'{s}_{a}' for s in ('out','in') for a in AGGREGATES]

def model():
    return HistGradientBoostingClassifier(max_iter=300,learning_rate=.05,random_state=0)

def logit(p):
    return torch.logit(torch.as_tensor(p,dtype=torch.float64).clamp(1e-6,1-1e-6))

def score(p,y,threshold):
    return compute_metrics(logit(p),torch.as_tensor(y),threshold)

def fit_seed(seed_dir,labels,cutoff):
    val=load(seed_dir/'scores_val','val',labels,cutoff);test=load(seed_dir/'scores','test',labels,cutoff)
    X,y=val[columns()].values,val.account_label.values
    # Threshold comes from out-of-fold validation predictions so it is not tuned on fitted rows.
    oof=np.zeros(len(y))
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(X,y):
        oof[te]=model().fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
    threshold=find_optimal_threshold(logit(oof),torch.as_tensor(y))
    stacker=model().fit(X,y)
    p=stacker.predict_proba(test[columns()].values)[:,1]
    base_threshold=json.loads((seed_dir/'results.json').read_text())['thresholds']['account_base']
    clean=~test.late.values;yt=test.account_label.values
    result={'threshold':threshold,'validation_oof':score(oof,y,threshold),
            'test':{'stacked':score(p,yt,threshold),'base':score(test.base_account_risk_score.values,yt,base_threshold)},
            'test_clean':{'stacked':score(p[clean],yt[clean],threshold),
                          'base':score(test.base_account_risk_score.values[clean],yt[clean],base_threshold)}}
    joblib.dump({'model':stacker,'columns':columns(),'threshold':threshold},seed_dir/'stacker.joblib')
    (seed_dir/'stacker.json').write_text(json.dumps(result,indent=2))
    for split in ('test','test_clean'):
        r=result[split];print(f"{seed_dir.name} {split}: base AP={r['base']['auprc']:.4f} stacked AP={r['stacked']['auprc']:.4f} "
                              f"stacked F1={r['stacked']['f1']:.4f}",flush=True)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--labels-file',type=Path,default=config.SPARK_STATIC_DIR/'account_labels.parquet')
    p.add_argument('--clean-after',default='2022-09-17',help='test_clean drops accounts touched at or after this UTC date')
    args=p.parse_args()
    seeds=sorted(d for d in args.run_dir.glob('seed_*') if (d/'scores_val').is_dir())
    if not seeds:raise FileNotFoundError('No seed_*/scores_val exports; train with --export-scores')
    labels=pd.read_parquet(args.labels_file,columns=['account_id','account_label'])
    cutoff=pd.Timestamp(args.clean_after,tz='UTC').timestamp()
    results=[fit_seed(d,labels,cutoff) for d in seeds]
    summary={split:{kind:aggregate_seed_metrics([r[split][kind] for r in results]) for kind in ('stacked','base')}
             for split in ('test','test_clean')}
    summary['seeds']=[d.name for d in seeds]
    (args.run_dir/'stacker_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
