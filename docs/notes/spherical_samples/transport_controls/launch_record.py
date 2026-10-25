import json,os,subprocess
from pathlib import Path
root=Path('/home/chengcli/scix/repos/snapy')
m=json.loads((root/'docs/notes/spherical_samples/eta098-n224-r8-coarse-f1000.csv.json').read_text())
for tag,distributed,tls in [('single',False,''),('ipc',True,'self,tcp,cuda_copy,cuda_ipc'),('copy',True,'self,tcp,cuda_copy')]:
 out=Path('/tmp/snapy-transport-'+tag+'.csv')
 command=[str(root/'build/bin/spherical_convection.release'),str(root/'examples/spherical_convection.yaml'),str(m['Ra']),'.98','111','224','8','.01','1e-6','.8','1','.5',str(out),'cuda:local' if distributed else 'cuda:0',str(root/'docs/notes/spherical_samples/eta098-n224-r8-coarse-f1000.profile'),'z','8','float64']
 env={**os.environ,'OPENBLAS_NUM_THREADS':'1'}
 if distributed:
  env.update(BACKEND='ucx',DEVICE='cuda',CUDA_VISIBLE_DEVICES='0,1',UCX_TLS=tls,COMMUX_GROUP='1',COMMUX_COALESCE='1')
  command=['torchrun','--standalone','--no-python','--nproc-per-node=2',*command]
 print('START',tag,flush=True)
 with out.with_suffix('.log').open('w') as log:
  subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
 metadata=json.loads(Path(str(out)+'.json').read_text())
 print('DONE',tag,metadata['seconds'],metadata['cycles'],flush=True)
