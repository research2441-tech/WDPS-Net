"""Build a unified multi-sensor HSI registry: each cube -> float32 [0,1] (H,W,B) + band-centre wavelengths (nm).
Wavelength tables: exact where the source provides them, otherwise nominal sensor grids minus documented band removals."""
import numpy as np, scipy.io as sio, h5py, os, json, tifffile
D = './data'; O = './cubes'; os.makedirs(O, exist_ok=True)

def norm(x, lo=0.1, hi=99.9):
    x = np.nan_to_num(x.astype(np.float32))
    a, b = np.percentile(x, lo), np.percentile(x, hi)
    return np.clip((x - a) / (b - a + 1e-8), 0, 1).astype(np.float32)

def keep(n, removed):  # removed: 1-based inclusive ranges
    r = set()
    for a, b in removed: r.update(range(a, b + 1))
    return np.array([i for i in range(n) if (i + 1) not in r])

def save(name, x, wl, sensor, role, exact, **kw):
    wl = np.asarray(wl, np.float32); assert x.shape[2] == len(wl), (name, x.shape, len(wl))
    np.savez_compressed(f'{O}/{name}.npz', x=x, wl=wl, **kw)
    meta[name] = dict(sensor=sensor, role=role, shape=list(x.shape), wl_min=float(wl.min()), wl_max=float(wl.max()), exact_wl=exact)

meta = {}
# AVIRIS 224-channel reference table from the Salton Sea ENVI header (exact AVIRIS centres for that flight)
ss = np.loadtxt(f'{D}/SaltonSea/salton_sea_wavelengths_nm.txt', comments='#', usecols=(1,)) if True else None
assert len(ss) == 224
ip = np.loadtxt(f'{D}/IndianPines/indian_pines_220_wavelengths_nm.txt', comments='#', usecols=(1,))

# ---------------- AVIRIS ----------------
m = sio.loadmat(f'{D}/JasperRidge/real_data_jasperRidge.mat'); x = m['r_cube']
save('jasper', norm(x), ss[keep(224, [(1, 3), (108, 112), (154, 166), (220, 224)])], 'AVIRIS', 'train', False)
m = sio.loadmat(f'{D}/TargetDetection/san_diego_ECEM.mat'); x = m['X'].T.reshape(200, 200, 189, order='C')
x2 = m['X'].reshape(189, 200, 200).transpose(2, 1, 0)
save('sandiego', norm(x2), ss[keep(224, [(1, 6), (33, 35), (97, 97), (107, 113), (153, 166), (221, 224)])], 'AVIRIS', 'train', False)
m = sio.loadmat(f'{D}/SaltonSea/salton_sea_roi.mat'); x = m['I']; good = ~np.isnan(x).any((0, 1))
save('saltonsea', norm(x[:, :, good]), ss[good], 'AVIRIS', 'train', True)
m = sio.loadmat(f'{D}/Salinas/Salinas_corrected.mat'); x = m['salinas_corrected']
save('salinas', norm(x), ss[keep(224, [(108, 112), (154, 167), (224, 224)])], 'AVIRIS', 'xscene', False,
     gt=sio.loadmat(f'{D}/Salinas/Salinas_gt.mat')['salinas_gt'])
m = sio.loadmat(f'{D}/TargetDetection/cuprite_ECEM.mat'); X = m['X']
x = X.reshape(188, 191, 250).transpose(2, 1, 0)
save('cuprite', norm(x), ss[keep(224, [(1, 2), (104, 113), (148, 167), (221, 224)])], 'AVIRIS', 'xscene', False,
     gt=m['groundtruth'].astype(np.uint8), target=m['d'].ravel().astype(np.float32))
# real-noise AVIRIS: Indian Pines raw 220 bands (water-absorption bands kept), exact wavelengths
x = sio.loadmat(f'{D}/IndianPines/Indian_pines.mat')['indian_pines']
save('indianpines220', norm(x), ip, 'AVIRIS', 'real', True, gt=sio.loadmat(f'{D}/IndianPines/Indian_pines_gt.mat')['indian_pines_gt'],
     noisy=np.array([i for i in range(220) if (104 <= i + 1 <= 108) or (150 <= i + 1 <= 163) or i + 1 == 220]))
x = sio.loadmat(f'{D}/Salinas/SalinasA/salinasA.mat')['salinasA']
save('salinasA224', norm(x), ss, 'AVIRIS', 'real', False, gt=sio.loadmat(f'{D}/Salinas/SalinasA/salinasA_gt.mat')['salinasA_gt'])

# ---------------- ROSIS ----------------
x = sio.loadmat(f'{D}/PaviaU/PaviaU.mat')['paviaU']
save('paviaU', norm(x), np.linspace(430, 860, 103), 'ROSIS', 'heldout', False, gt=sio.loadmat(f'{D}/PaviaU/PaviaU_gt.mat')['paviaU_gt'])
m = sio.loadmat(f'{D}/AnomalyDetection_ABU/pavia.mat')
save('paviaC', norm(m['data']), np.linspace(430, 860, 102), 'ROSIS', 'heldout', False)

# ---------------- HYDICE ----------------
m = sio.loadmat(f'{D}/Urban_HYDICE/Urban6.mat'); Y = m['Y'].astype(np.float32)
x = Y.reshape(162, 307, 307).transpose(2, 1, 0)
hy = np.linspace(400, 2500, 210)
save('urban', norm(x), hy[keep(210, [(1, 4), (76, 76), (87, 87), (101, 111), (136, 153), (198, 210)])], 'HYDICE', 'heldout', False)
x = sio.loadmat(f'{D}/WashingtonDC/Washington-crop-test.mat')['temp']
save('dcmall', norm(x), np.linspace(400, 2400, 191), 'HYDICE', 'heldout', False)

# ---------------- Hyperion ----------------
x = sio.loadmat(f'{D}/Botswana/Botswana.mat')['Botswana']
hyp = np.array([355.59 + 10.175 * i if i < 70 else 851.92 + 10.09 * (i - 70) for i in range(242)])
kb = np.concatenate([np.arange(a - 1, b) for a, b in [(10, 55), (82, 97), (102, 119), (134, 164), (187, 220)]])
save('botswana', norm(x), hyp[kb], 'Hyperion', 'heldout', False, gt=sio.loadmat(f'{D}/Botswana/Botswana_gt.mat')['Botswana_gt'])

# ---------------- EnMAP (real noise, spaceborne) ----------------
t = tifffile.imread(f'{D}/EnMAP_Potsdam/enmap_potsdam.tif').astype(np.float32)
t[t == -32768] = np.nan; t = t.transpose(1, 2, 0)
wlE = np.loadtxt(f'{D}/EnMAP_Potsdam/enmap_potsdam_wavelengths_nm.txt', comments='#', usecols=(1,))
save('enmap', norm(t), wlE, 'EnMAP', 'real', True, noisy=np.arange(129, 135))

# ---------------- CASI / AISA (VNIR airborne) ----------------

wl72 = np.loadtxt(f'{D}/MUUFL_Gulfport_CASI/muufl_72band_wavelengths_nm.txt', comments='#', usecols=(1,))
m = sio.loadmat(f'{D}/MUUFL_Gulfport_CASI/muufl_gulfport_campus_1_hsi_220_label.mat', squeeze_me=True, struct_as_record=False)['hsi']
k64 = keep(72, [(1, 4), (69, 72)])
save('muufl64', norm(m.Data), wl72[k64], 'CASI', 'heldout', True, gt=m.sceneLabels.labels.astype(np.int16))
# real-noise CASI: full 72-band cube incl. noisy edge bands + emplaced-target ground truth
m = sio.loadmat(f'{D}/MUUFL_Gulfport_CASI/muufl_gulfport_campus_w_lidar_1.mat', squeeze_me=True, struct_as_record=False)['hsi']
g = m.groundTruth
save('muufl72', norm(m.Data), wl72, 'CASI', 'real', True, noisy=np.array([0, 1, 2, 3, 68, 69, 70, 71]),
     t_row=np.asarray(g.Targets_rowIndices), t_col=np.asarray(g.Targets_colIndices),
     t_type=np.asarray([str(v) for v in g.Targets_Type]), t_size=np.asarray(g.Targets_Size, float), valid=m.valid_mask.astype(np.uint8))
for nm in ['Houston13', 'Houston18']:
    f = h5py.File(f'{D}/Houston_CASI/{nm}.mat', 'r'); x = np.array(f['ori_data']).transpose(2, 1, 0)
    save(nm.lower(), norm(x), np.linspace(380, 1050, 48), 'CASI', 'heldout', False)
x = sio.loadmat(f'{D}/Trento_AISA/Italy_hsi.mat')['data']
save('trento', norm(x), np.linspace(402.89, 989.09, 63), 'AISA', 'heldout', False)
# ---------------- always-train extra sensors ----------------
x = sio.loadmat(f'{D}/APEX/APEX_Y_clean.mat')['Y_clean']
save('apex', norm(x), np.linspace(413, 2420, 285), 'APEX', 'train', False)
m = sio.loadmat(f'{D}/Samson/Samson.mat'); x = m['Y'].reshape(156, 95, 95).transpose(2, 1, 0)
save('samson', norm(x), np.linspace(401, 889, 156), 'SAMSON', 'train', False)
import spectral
im = spectral.envi.open(f'{D}/HyLite_test/image.hdr', f'{D}/HyLite_test/image.dat'); x = np.asarray(im.load())
save('fenix', norm(x), np.array(im.bands.centers), 'FENIX', 'train', True)
# ---------------- natural-scene sensors ----------------
x = sio.loadmat(f'{D}/CAVE/fake_and_real_peppers_ms.mat')['hrhsi']
save('cave_peppers', norm(x), np.arange(400, 701, 10), 'CAVE', 'heldout', True)
x = sio.loadmat(f'{D}/CAVE/pompoms_ms.mat')['truth'].reshape(31, 512, 512).transpose(2, 1, 0)
save('cave_pompoms', norm(x), np.arange(400, 701, 10), 'CAVE', 'heldout', True)
kwl = [453.3, 457.6, 462.1, 466.8, 471.6, 476.5, 481.6, 486.9, 492.4, 498.0, 503.9, 509.9, 516.2, 522.7, 529.5, 536.5,
       543.8, 551.4, 558.6, 567.5, 575.3, 584.3, 594.4, 604.2, 614.4, 625.1, 636.3, 648.1]
for i in range(1, 11):
    x = sio.loadmat(f'{D}/KAIST_TSA/scene{i:02d}.mat')['img']
    save(f'kaist{i:02d}', norm(x), kwl, 'KAIST', 'heldout', False)
json.dump(meta, open(f'{O}/meta.json', 'w'), indent=1)
for k, v in meta.items(): print(k, v['sensor'], v['role'], v['shape'], round(v['wl_min']), round(v['wl_max']))
