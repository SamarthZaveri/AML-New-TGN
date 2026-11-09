"""Joint account/transaction experiments. Run manually; existing trainer stays intact."""
import argparse,json,random,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import config
from stage2.transaction_dataset import TransactionDataset
from stage2.transaction_model import TransactionAccountTGN
from stage2.conservation_loss import FocalLoss
from stage2.evaluate import compute_metrics,find_optimal_threshold,aggregate_seed_metrics
from stage2.checkpointing import StopRequest,atomic_save,load_trusted,rng_state,restore_rng

FORMAT=1
class Paused(Exception):pass

def initialize(model,dataset,device):
    model.initialize(dataset.account_features.to(device),dataset.bank_features.to(device),
                     dataset.entity_features.to(device),dataset.account_bank_map.to(device),dataset.account_entity_map.to(device))

def invoke(model,batch,relay,alpha,score_accounts=True):
    return model(**{k:batch[k] for k in ('source_ids','dest_ids','edge_features','timestamps')},
                 do_relay=relay,merge_alpha=alpha,score_accounts=score_accounts)

def metrics(logits,labels,threshold=None):
    # Preserve explicit eligibility problems instead of returning misleading zero metrics.
    if len(labels)==0 or len(torch.unique(labels))!=2:raise ValueError('Evaluation needs both label classes')
    if threshold is None:threshold=find_optimal_threshold(logits,labels)
    return {'at_0_5':compute_metrics(logits,labels,.5),'at_validation_threshold':compute_metrics(logits,labels,threshold)},threshold

@torch.no_grad()
def evaluate(model,dataset,args,stop,target,thresholds=None,export_dir=None):
    model.eval();initialize(model,dataset,args.device)
    n=dataset.num_accounts
    base=torch.zeros(n); final=torch.zeros(n); active=torch.zeros(n,dtype=torch.bool)
    edge_logits=[];edge_labels=[];step=0
    writer=None
    if export_dir:
        import pyarrow as pa
        import pyarrow.parquet as pq
        export_dir.mkdir(parents=True,exist_ok=True)
    windows=('train','val') if target=='val' else ('train','val','test')
    try:
        for window in windows:
            for b in range(dataset.num_batches(window)):
                if stop.requested:raise Paused('Evaluation interrupted; resume replays evaluation from training checkpoint')
                batch=dataset.get_batch(b,args.device,split=window)
                out=invoke(model,batch,step%args.relay_interval==0,args.merge_alpha,score_accounts=window==target);step+=1
                checked=('base_account_logits','final_account_logits','transaction_logits') if window==target else ('transaction_logits',)
                if not all(torch.isfinite(out[k]).all() for k in checked):
                    raise FloatingPointError('Non-finite evaluation outputs')
                if window!=target:continue
                ids=out['node_ids'].cpu();base[ids]=out['base_account_logits'].cpu();final[ids]=out['final_account_logits'].cpu();active[ids]=True
                edge_logits.append(out['transaction_logits'].cpu());edge_labels.append(batch['transaction_labels'].cpu())
                if export_dir:
                    source=batch['source_ids'].cpu().numpy();dest=batch['dest_ids'].cpu().numpy()
                    table=pa.Table.from_pydict({'event_idx':batch['original_event_ids'].numpy(),
                       'sender_account_id':dataset.account_index[source].astype(str).tolist(),
                       'receiver_account_id':dataset.account_index[dest].astype(str).tolist(),
                       'timestamp_unix':batch['timestamps'].cpu().numpy()*3600+dataset.time_origin,
                       'transaction_risk_score':out['transaction_logits'].sigmoid().cpu().numpy(),
                       'transaction_label':batch['transaction_labels'].cpu().numpy()})
                    if writer is None:writer=pq.ParquetWriter(export_dir/'transaction_scores.parquet',table.schema)
                    writer.write_table(table)
            print(f'Replayed {window}: {step:,} batches',flush=True)
    finally:
        if writer is not None:writer.close()
    mask=getattr(dataset,target+'_mask') & active
    y=dataset.account_labels[mask]
    el=torch.cat(edge_logits);ey=torch.cat(edge_labels)
    results={};chosen={}
    for name,logits,labels in [('account_base',base[mask],y),('account_final',final[mask],y),('transaction',el,ey)]:
        results[name],chosen[name]=metrics(logits,labels,None if thresholds is None else thresholds[name])
    results['eligible_accounts']=int(mask.sum());results['transactions']=len(ey)
    if export_dir:
        ids=active.nonzero(as_tuple=True)[0]
        pd.DataFrame({'account_id':dataset.account_index[ids.numpy()].astype(str),
             'base_account_risk_score':base[ids].sigmoid().numpy(),'risk_score':final[ids].sigmoid().numpy(),
             f'{target}_account_eligible':mask[ids].numpy()}).to_parquet(export_dir/'account_scores.parquet',index=False)
        (export_dir/'manifest.json').write_text(json.dumps({'scope':f'{target}-window transaction predictions and last-touch active-account scores',
            'mode':args.mode,'fingerprint':dataset.fingerprint,'thresholds':chosen},indent=2))
    return results,chosen

def train_seed(dataset,args,seed,stop):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    model_args={'num_account_nodes':dataset.num_accounts,'num_bank_nodes':dataset.num_banks,'num_entity_nodes':dataset.num_entities,
                'account_feat_dim':dataset.account_features.size(1),'bank_feat_dim':dataset.bank_features.size(1),
                'entity_feat_dim':dataset.entity_features.size(1),'edge_feat_dim':dataset.edge_feat_dim,
                'memory_dim':args.hidden_dim,'time_dim':args.time_dim,'num_attention_heads':args.num_heads,
                'dropout':args.dropout,'enable_relay':not args.disable_relay,'max_neighbors':args.max_neighbors}
    model=TransactionAccountTGN(mode=args.mode,**model_args).to(args.device)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=args.lr,weight_decay=args.weight_decay)
    account_loss=FocalLoss(args.focal_gamma,dataset.compute_pos_weight(args.max_account_pos_weight))
    transaction_loss=FocalLoss(args.focal_gamma,dataset.transaction_pos_weight(args.max_transaction_pos_weight))
    settings={k:v for k,v in vars(args).items() if k not in ('resume','run_dir','num_seeds','seeds','events_file','static_dir','chain_signals_file')}
    settings.update(model=model_args,seed=seed)
    path=args.run_dir/f'seed_{seed}';path.mkdir(parents=True,exist_ok=True)
    latest,best=path/'latest.pt',path/'best.pt'
    progress={'epoch':0,'next_batch':0,'best_ap':-1.,'best_epoch':0,'stale':0,'history':[],'loss_sum':0.,'loss_batches':0}
    initialize(model,dataset,args.device)
    resuming=args.resume and latest.exists()  # seeds not yet started under --resume begin fresh
    if resuming:
        checkpoint=load_trusted(latest,'cpu')
        if checkpoint.get('joint_format')!=FORMAT or checkpoint['fingerprint']!=dataset.fingerprint or checkpoint['settings']!=settings:
            raise ValueError('Resume requires identical dataset, architecture, TBPTT window and experiment settings')
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer']);progress=checkpoint['progress'];restore_rng(checkpoint['rng'])
        del checkpoint
    elif latest.exists() or best.exists():raise FileExistsError('Run already exists; choose a new directory or --resume')
    def save():
        model.detach_window()
        atomic_save({'joint_format':FORMAT,'fingerprint':dataset.fingerprint,'settings':settings,
                     'dataset_metadata':dataset.metadata,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                     'progress':progress,'rng':rng_state()},latest)
    if not resuming:save()
    nb=dataset.num_batches('train');labels=dataset.account_labels.to(args.device);training=dataset.train_mask.to(args.device)
    last_save=time.monotonic()
    while progress['epoch']<args.num_epochs and progress['stale']<args.patience:
        if progress['next_batch']==0:initialize(model,dataset,args.device)
        model.train();b=progress['next_batch']
        while b<nb:
            if stop.requested:save();raise Paused('Stopped at optimizer-window boundary')
            optimizer.zero_grad(set_to_none=True);losses=[]
            end=min(b+args.bptt_batches,nb)
            for index in range(b,end):
                batch=dataset.get_batch(index,args.device,split='train')
                out=invoke(model,batch,index%args.relay_interval==0,args.merge_alpha)
                ids=out['node_ids'];mask=training[ids]
                chosen=out['base_account_logits'] if args.mode=='separate' else out['final_account_logits']
                primary=account_loss(chosen[mask],labels[ids[mask]])
                edge=transaction_loss(out['transaction_logits'],batch['transaction_labels'])
                loss=primary+args.lambda_transaction*edge
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss; recover from latest window checkpoint')
                losses.append(loss)
                progress['loss_sum']+=float(loss.detach());progress['loss_batches']+=1
            torch.stack(losses).mean().backward()
            nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            optimizer.step();model.detach_window();b=end;progress['next_batch']=b
            if b%100==0 or b==nb:print(f'Seed {seed} epoch {progress["epoch"]+1} batch {b}/{nb}: loss={float(loss.detach()):.5f}',flush=True)
            if time.monotonic()-last_save>=args.checkpoint_seconds or stop.requested:
                save();last_save=time.monotonic()
            if stop.requested:raise Paused('Saved after completed TBPTT optimizer window')
        save()  # authoritative checkpoint if replay is interrupted
        val,thresholds=evaluate(model,dataset,args,stop,'val')
        # Separate mode trains only the base head; the final score is an untrained fixed blend.
        ap=val['account_base' if args.mode=='separate' else 'account_final']['at_0_5']['auprc']
        epoch=progress['epoch']+1
        if ap>progress['best_ap']:
            progress.update(best_ap=ap,best_epoch=epoch,stale=0)
            atomic_save({'joint_format':FORMAT,'kind':'joint_inference','fingerprint':dataset.fingerprint,
                'dataset_metadata':dataset.metadata,'settings':settings,'epoch':epoch,'thresholds':thresholds,
                'weights':{name:p.detach().cpu() for name,p in model.named_parameters()},'validation':val},best)
        else:progress['stale']+=1
        progress['history'].append({'epoch':epoch,'loss':progress['loss_sum']/max(progress['loss_batches'],1),'validation':val})
        print(json.dumps({'epoch':epoch,'validation':val}),flush=True)
        progress.update(epoch=epoch,next_batch=0,loss_sum=0.,loss_batches=0)
        initialize(model,dataset,args.device);save()
        # Release blocks cached by the evaluation replay so training doesn't hit fragmentation OOMs.
        if torch.cuda.is_available():torch.cuda.empty_cache()
        (path/'history.json').write_text(json.dumps(progress['history'],indent=2))
    checkpoint=load_trusted(best,'cpu')
    with torch.no_grad():
        for name,p in model.named_parameters():p.copy_(checkpoint['weights'][name])
    thresholds=checkpoint['thresholds']
    # Validation-window scores are the training data for the post-hoc stacker.
    if args.export_scores:evaluate(model,dataset,args,stop,'val',thresholds,path/'scores_val')
    test,_=evaluate(model,dataset,args,stop,'test',thresholds,path/'scores' if args.export_scores else None)
    result={'seed':seed,'mode':args.mode,'best_epoch':checkpoint['epoch'],'test':test,'thresholds':thresholds,'dataset_fingerprint':dataset.fingerprint}
    (path/'results.json').write_text(json.dumps(result,indent=2));return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['separate','fused'],required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--events-file',type=Path,default=config.EVENT_STREAM_FILE)
    p.add_argument('--static-dir',type=Path,default=config.SPARK_STATIC_DIR)
    p.add_argument('--chain-signals-file',type=Path,default=config.CHAIN_SIGNALS_FILE)
    p.add_argument('--num-seeds',type=int,default=1);p.add_argument('--num-epochs',type=int,default=20)
    p.add_argument('--seeds',type=int,nargs='+',help='explicit model seeds; overrides --num-seeds (data split stays fixed)')
    p.add_argument('--batch-size',type=int,default=1000);p.add_argument('--bptt-batches',type=int,default=1)
    p.add_argument('--hidden-dim',type=int,default=128);p.add_argument('--time-dim',type=int,default=64)
    p.add_argument('--num-heads',type=int,default=4);p.add_argument('--max-neighbors',type=int,default=20)
    p.add_argument('--dropout',type=float,default=.1);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--weight-decay',type=float,default=1e-5);p.add_argument('--focal-gamma',type=float,default=2.)
    p.add_argument('--lambda-transaction',type=float,default=1.);p.add_argument('--merge-alpha',type=float,default=.5)
    p.add_argument('--max-account-pos-weight',type=float,default=50.);p.add_argument('--max-transaction-pos-weight',type=float,default=500.)
    p.add_argument('--relay-interval',type=int,default=1);p.add_argument('--disable-relay',action='store_true')
    p.add_argument('--patience',type=int,default=5);p.add_argument('--checkpoint-seconds',type=float,default=600.)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--resume',action='store_true');p.add_argument('--export-scores',action='store_true')
    args=p.parse_args()
    for name in ('num_seeds','num_epochs','batch_size','bptt_batches','hidden_dim','time_dim','num_heads','max_neighbors','relay_interval','patience'):
        if getattr(args,name)<1:p.error(name+' must be positive')
    if args.hidden_dim<2 or args.hidden_dim%args.num_heads:p.error('hidden-dim must be >=2 and divisible by num-heads')
    if not 0<=args.merge_alpha<=1 or not 0<=args.dropout<1:p.error('Invalid alpha/dropout')
    if args.lr<=0 or args.lambda_transaction<=0 or args.checkpoint_seconds<=0 or args.max_account_pos_weight<=0 or args.max_transaction_pos_weight<=0:p.error('Invalid loss/optimizer/checkpoint settings')
    if args.weight_decay<0 or args.focal_gamma<0:p.error('Negative weight decay/focal gamma')
    if args.seeds is not None and len(set(args.seeds))!=len(args.seeds):p.error('Duplicate --seeds')
    seeds=args.seeds or [42+111*i for i in range(args.num_seeds)]
    dataset=TransactionDataset(args.events_file,args.static_dir,args.chain_signals_file,args.batch_size,split_mode='chronological')
    args.run_dir.mkdir(parents=True,exist_ok=True)
    (args.run_dir/'experiment.json').write_text(json.dumps({'settings':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'dataset_metadata':dataset.metadata,'fingerprint':dataset.fingerprint},indent=2))
    results=[]
    with StopRequest() as stop:
        try:
            for seed in seeds:results.append(train_seed(dataset,args,seed,stop))
        except Paused as exc:
            print(f'{exc}. Resume with the identical command plus --resume.',flush=True);return
    summary={'runs':results,'aggregate':{task:aggregate_seed_metrics([row['test'][task]['at_validation_threshold'] for row in results]) for task in ('account_base','account_final','transaction')}}
    (args.run_dir/'summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
