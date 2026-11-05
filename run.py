#!/usr/bin/env python3
"""Run Full/Basic models, their ablations, Mixed-CBM interventions, and the selected-concept CBM."""
import argparse
import fcntl
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
import yaml

HERE=Path(__file__).resolve().parent

def make_reports(cfg, concepts_only=False):
    from infosign.report import make_report
    from infosign.cbm.report import make_diagnostics_report, make_recovery_report
    if not concepts_only:
        make_report(cfg)
    make_diagnostics_report(cfg)
    make_recovery_report(cfg)

def run_worker(cfg, task, phase, cpu=False):
    """Execute one scheduled job on its assigned visible GPU."""
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[key]=str(cfg['cpu_threads'])
    import torch
    torch.set_num_threads(cfg['cpu_threads'])
    torch.set_float32_matmul_precision('high')
    device=torch.device('cpu' if cpu else 'cuda:0')
    if device.type=='cuda':torch.backends.cudnn.benchmark=True
    if phase=='concept-diagnostics':
        from infosign.cbm.diagnostics import run_diagnostics
        run_diagnostics(task,cfg,device)
    elif phase=='signature':
        from infosign.cbm.signature import run_signature
        run_signature(task,cfg,device)
    elif phase=='analyze':
        from infosign.analysis import run_analysis
        run_analysis(task,cfg,device)
    else:
        from infosign.train import run_model
        if not cfg.get('backbone_root'):
            run_model(task,phase,cfg,device)
        if device.type=='cuda':torch.cuda.empty_cache()
        if phase=='full':
            from infosign.cbm.model import run_cbm
            run_cbm(task,cfg,device)

def pool(jobs,devices,config,root,after=None):
    waiting=list(jobs);free=list(map(str,devices));running={};failed=[];selector=selectors.DefaultSelector()
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    def launch(job,gpu):
        name,args=job;env=os.environ.copy();cpu=gpu=='cpu'
        if not cpu:env['CUDA_VISIBLE_DEVICES']=visible.split(',')[int(gpu)] if visible else gpu
        command=[sys.executable,'-u',str(HERE/'run.py'),'--config',str(config),*args]+(['--cpu'] if cpu else [])
        log=(root/'logs'/(name+'.log')).open('ab',buffering=0)
        proc=subprocess.Popen(command,cwd=HERE,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              bufsize=0,start_new_session=True)
        running[name]={'proc':proc,'gpu':gpu,'log':log,'buffer':b'','eof':False}
        selector.register(proc.stdout,selectors.EVENT_READ,name)
        print(f'[start] {name} on GPU {gpu} (PID {proc.pid})',flush=True)
    try:
        last=time.monotonic()
        while waiting or running:
            while waiting and free:launch(waiting.pop(0),free.pop(0))
            events=selector.select(1) if selector.get_map() else []
            if not events:time.sleep(.05)
            for key,_ in events:
                name=key.data;item=running[name];chunk=os.read(key.fd,65536)
                if not chunk:selector.unregister(key.fileobj);item['eof']=True;continue
                item['log'].write(chunk);item['buffer']+=chunk
                while b'\n' in item['buffer']:
                    line,item['buffer']=item['buffer'].split(b'\n',1)
                    print('['+name+'] '+line.decode(errors='replace'),flush=True)
            for name,item in list(running.items()):
                code=item['proc'].poll()
                if code is None or not item['eof']:continue
                if item['buffer']:print(item['buffer'].decode(errors='replace'),flush=True)
                if code:failed.append(name)
                print(f'[finished] {name}, exit={code}',flush=True)
                item['log'].close();item['proc'].stdout.close();free.append(item['gpu']);del running[name]
                if after:after()
            if time.monotonic()-last>30:
                print('[running] '+', '.join(running)+f'; queued={len(waiting)}',flush=True);last=time.monotonic()
        if failed:raise RuntimeError('Failed units: '+', '.join(failed)+'. Fix the reported error and rerun the same command; completed units are reused.')
    finally:
        for item in running.values():
            proc=item['proc']
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            item['log'].close();proc.stdout.close()
        selector.close()

def main():
    p=argparse.ArgumentParser(description='Six-task multimodal and CBM study from prepared inputs.')
    p.add_argument('--config',type=Path,default=HERE/'config.yaml')
    p.add_argument('--devices',nargs='+');p.add_argument('--source-root',type=Path)
    p.add_argument('--output-root',type=Path)
    p.add_argument('--stage',choices=('all','train','analyze','report','concept-diagnostics','concept-diagnostics-report','signature','signature-report'),default='all')
    p.add_argument('--worker',choices=('full','basic','analyze','concept-diagnostics','signature'),help=argparse.SUPPRESS)
    p.add_argument('--task',help=argparse.SUPPRESS)
    p.add_argument('--cpu',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args();settings=yaml.safe_load(a.config.read_text())
    cfg=settings.get('study',settings)
    if a.worker:
        if not a.task:p.error('--worker requires --task')
        run_worker(cfg,a.task,a.worker,a.cpu)
        return
    for key in ('source_root','output_root','devices'):
        if getattr(a,key) is not None:cfg[key]=getattr(a,key)
    for key in ('source_root','output_root','clinical_dir'):
        cfg[key]=str(Path(cfg[key]).expanduser().resolve())
    if cfg.get('backbone_root'):
        cfg['backbone_root']=str(Path(cfg['backbone_root']).expanduser().resolve())
        if cfg['backbone_root']==cfg['output_root']:
            raise ValueError('Choose a separate output-root for the new CBMs.')
    cfg['devices']=list(map(str,cfg['devices']))
    root=Path(cfg['output_root']);source=Path(cfg['source_root'])
    if root==source:raise ValueError('Use separate prepared-input and experiment output directories.')
    root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]=str(cfg['cpu_threads'])
    from infosign.common import read_json,write_json
    from infosign.report import make_report
    diagnostics = a.stage in ('concept-diagnostics','concept-diagnostics-report')
    signature = a.stage == 'signature'
    runtime=root/'runtime.yaml'
    if runtime.exists():
        previous=yaml.safe_load(runtime.read_text())
        # Preserve a resumed experiment; allow only scheduling resources to change.
        resources=('devices','num_workers','cpu_threads')
        if {k:v for k,v in previous.items() if k not in resources}!={k:v for k,v in cfg.items() if k not in resources}:
            raise ValueError('Settings differ from this output. Use its runtime.yaml or choose a new output-root.')
    if a.stage in ('report','concept-diagnostics-report'):
        make_reports(cfg,concepts_only=diagnostics)
        return
    if a.stage == 'signature-report':
        from infosign.cbm.signature import make_signature_report
        make_signature_report(cfg)
        return
    with (root/'.run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('A process is already using this output directory') from None
        runtime.write_text(yaml.safe_dump(cfg,sort_keys=False))
        status_path = root/('CONCEPT_DIAGNOSTICS_STATUS.json' if diagnostics else 'SIGNATURE_STATUS.json' if signature else 'STATUS.json')
        status=dict(status='running',stage=a.stage,tasks=cfg['tasks']);write_json(status_path,status)
        try:
            if a.stage in ('all','train'):
                # These are existing normalized tensors and frozen ResNet features,
                # not fitted representations from the previous experiment.
                if not (source/'inputs.json').exists() or not (source/'cache/dimensions.json').exists():
                    raise FileNotFoundError('Prepared inputs/cache are missing at '+str(source)+
                        '. Download AI-READI separately, then run prepare.py; see README.md.')
                read_json(source/'inputs.json');read_json(source/'cache/dimensions.json')
                if not (root/'labels/DONE.json').exists():
                    from infosign.cbm.labels import extract,FOUR_OBSERVATION_FORMS,SURVEY_ADMIN_IDS
                    extract(cfg['clinical_dir'],root/'labels',observation_forms=() if cfg.get('exclude_observation') else FOUR_OBSERVATION_FORMS,excluded_ids=SURVEY_ADMIN_IDS,tasks=cfg['tasks'])
                    write_json(root/'labels/DONE.json',dict(clinical_dir=cfg['clinical_dir']))
                jobs=[(task+'_'+variant,['--task',task,'--worker',variant]) for task in cfg['tasks'] for variant in ('full','basic')]
                pool(jobs,cfg['devices'],runtime,root)
            if a.stage in ('all','analyze'):
                jobs=[(task+'_analysis',['--task',task,'--worker','analyze']) for task in cfg['tasks']]
                pool(jobs,cfg['devices'],runtime,root,after=lambda:make_report(cfg))
            if diagnostics or a.stage in ('all','analyze'):
                jobs=[(task+'_concept_diagnostics',['--task',task,'--worker','concept-diagnostics']) for task in cfg['tasks']]
                pool(jobs,cfg['devices'],runtime,root,after=lambda:make_reports(cfg,concepts_only=True))
            if a.stage in ('all','signature'):
                from infosign.cbm.signature import make_signature_report
                missing=[t for t in cfg['tasks'] if not (root/t/'concept_diagnostics'/'restoration.csv').exists()
                         or not (root/t/'mixed_cbm'/'model.pt').exists()]
                if missing:
                    raise FileNotFoundError('Selected-concept CBM requires concept diagnostics and the Mixed CBM for: '+', '.join(missing))
                jobs=[(task+'_signature',['--task',task,'--worker','signature']) for task in cfg['tasks']]
                pool(jobs,cfg['devices'],runtime,root,after=lambda:make_signature_report(cfg))
            status['status']='complete';write_json(status_path,status)
            if a.stage in ('all','signature'):
                from infosign.cbm.signature import make_signature_report
                make_signature_report(cfg)
            if not signature:
                make_reports(cfg,concepts_only=diagnostics)
        except BaseException as exc:
            status.update(status='failed',error=str(exc));write_json(status_path,status);raise
    print(('CONCEPT_DIAGNOSTICS_COMPLETE: ' if diagnostics else 'SIGNATURE_COMPLETE: ' if signature else 'STUDY_COMPLETE: ')+
          str(root/('tables/concept_diagnostics' if diagnostics else 'tables')),flush=True)

if __name__=='__main__':main()
