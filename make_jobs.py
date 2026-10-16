import wdps_data as D
FOLDS = list(D.FOLDS)
MODELS = ['WDPS', 'SSMBase', 'QRNN3D', 'SSTLite', 'PromptIR3D', 'SqueezeHSI']
R = './results'; E = 'WDPS_THREADS=1 OMP_NUM_THREADS=1'
def job(model, fold, seed, tag='', extra=''):
    name = f"{model}{tag}_{fold.replace('/', '-')}_s{seed}"
    return (f"{R}/{name}.json ::: {E} python3 train.py --model {model} --fold '{fold}' --seed {seed} --steps 800 --tag='{tag}' {extra} "
            f"> logs/{name}.log 2>&1 && {E} python3 evaluate.py ckpt/{name}.pt >> logs/{name}.log 2>&1")
L = []
PRE_ABL = {'-wave': '--wave index', '-gap': '--no_gap', '-kappa1': '--gap_learned 0', '-kappa1': '--gap_learned 0', '-prompt': '--no_prompt --lam_ctr 0', '-route': '--no_route', '-subspace': '--no_subspace', '-specSSM': '--no_spec'}
for tag, ex in PRE_ABL.items(): L.append(job('WDPS', 'ROSIS', 0, tag, ex))
for seed in range(3):
    for fold in FOLDS:
        for m in MODELS: L.append(job(m, fold, seed))
    if seed == 0:
        for c in ['LRMR', 'FastHyDe', 'BM4D', 'L1HyMixDe', 'Noisy']:
            L.append(f"{R}/{c}_classical.json ::: OMP_NUM_THREADS=1 python3 -W ignore classical.py {c} > logs/cl_{c}.log 2>&1")
ABL = {'-wave': '--wave index', '-gap': '--no_gap', '-kappa1': '--gap_learned 0', '-prompt': '--no_prompt --lam_ctr 0', '-route': '--no_route', '-subspace': '--no_subspace', '-specSSM': '--no_spec'}
for fold in ['ROSIS', 'HYDICE']:
    for tag, ex in ABL.items(): L.append(job('WDPS', fold, 0, tag, ex))
SENS = {'_K2': '--K 2', '_K6': '--K 6', '_F4': '--freqs 4', '_F16': '--freqs 16'}

E1 = 'WDPS_THREADS=1 OMP_NUM_THREADS=1'
for fold in FOLDS:
    f = fold.replace('/', '-')
    for seed in range(3):
        for m in (['WDPS', 'QRNN3D'] if seed == 0 else []):
            L.append(f"{R}/{m}_{f}_s{seed}+SSA.json ::: {E1} python3 adapt.py ckpt/{m}_{f}_s{seed}.pt 150 > logs/ssa_{m}_{f}_{seed}.log 2>&1")
    L.append(f"{R}/WDPS_{f}_s0+SSA-um.json ::: {E1} python3 adapt.py ckpt/WDPS_{f}_s0.pt 150 0 > logs/ssa_um_{f}.log 2>&1")
for m in ['WDPS', 'SSMBase', 'QRNN3D', 'SSTLite', 'PromptIR3D', 'SqueezeHSI']:
    L.append(f"{R}/downstream_{m}.json ::: {E1} python3 downstream.py {m} 'ckpt/{m}_ROSIS_s?.pt,ckpt/{m}_VNIR-CASI-AISA_s?.pt' > logs/ds_{m}.log 2>&1")
    L.append(f"{R}/robust_{m}.json ::: {E1} python3 analysis_runs.py robust {m} > logs/rob_{m}.log 2>&1")
for c in ['Noisy', 'FastHyDe', 'LRMR']:
    L.append(f"{R}/downstream_{c}.json ::: OMP_NUM_THREADS=1 python3 -W ignore downstream.py {c} classical > logs/ds_{c}.log 2>&1")
    L.append(f"{R}/robust_{c}.json ::: OMP_NUM_THREADS=1 python3 -W ignore analysis_runs.py robust {c} > logs/rob_{c}.log 2>&1")
for a in ['sens', 'routing', 'ensemble']:
    L.append(f"{R}/{a}_WDPS.json ::: {E1} python3 analysis_runs.py {a} > logs/an_{a}.log 2>&1")
W = [l for l in L if '--model WDPS' in l or 'adapt.py ckpt/WDPS' in l]
WM = [l for l in W if "--tag=''" in l]
L = WM + [l for l in L if l not in W] + [l for l in W if l not in WM]
open('jobs.txt', 'w').write('\n'.join(L) + '\n'); print(len(L), 'jobs')
