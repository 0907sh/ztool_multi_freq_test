"""
Compare single-frequency vs multi-frequency scan: time and admittance accuracy

- single_freq: 원본 ztoolacdc (site-packages), multi_freq_scan=False
- multi_freq:  수정본 ztoolacdc_mf (GitHub), multi_freq_scan=True (AC active scan 지원)

설치:
    pip install ztoolacdc==0.1.28                                          # 원본
    pip install git+https://github.com/0907sh/ztool_multi_freq_test.git   # 수정본
"""
import os
import time
import numpy as np
import ztoolacdc_mf.frequency_sweep as mf_frequency_sweep
import ztoolacdc_mf.create_freq     as mf_create_freq
import ztoolacdc_mf.read_admittance as mf_read_admittance
import ztoolacdc_mf.stability       as mf_stability

import ztoolacdc_mf
print("ztoolacdc_mf:", ztoolacdc_mf.__file__)

script_dir     = os.path.dirname(os.path.abspath(__file__))
pscad_folder   = script_dir + '\\'
results_folder = script_dir + r'\Results'
workspace_name = "Single_bus_example"
project_name   = "Simple_2L_VSC_RLC"

f_points      = 8 * 50
f_base        = 0.5
f_min         = 1.0
f_max         = 500.0
start_fft     = 1.0
fft_periods   = 1
dt_injections = 1
t_snap        = 10
t_sim         = start_fft + fft_periods / f_base
t_step        = 20.0
v_perturb_mag = 0.02

freq = mf_create_freq.loglist(f_min=f_min, f_max=f_max, f_points=f_points, f_base=f_base)

common = dict(
    t_snap=t_snap, t_sim=t_sim, t_step=t_step, dt_injections=dt_injections,
    f_base=f_base, freq=freq, start_fft=start_fft, fft_periods=fft_periods,
    v_perturb_mag=v_perturb_mag, working_dir=pscad_folder,
    workspace_name=workspace_name, project_name=project_name,
    results_folder=results_folder,
)

# ── Run 1: 원본 라이브러리, single_freq ──────────────────────────────────────
t0 = time.time()
mf_frequency_sweep.frequency_sweep(**common, output_files='single_freq', multi_freq_scan=False)
t_single = time.time() - t0
print(f"\n[single_freq] elapsed: {t_single:.1f} s")

# ── Run 2: 수정본 라이브러리, multi_freq ─────────────────────────────────────
t0 = time.time()
mf_frequency_sweep.frequency_sweep(**common, output_files='multi_freq', multi_freq_scan=True)
t_multi = time.time() - t0
print(f"[multi_freq]  elapsed: {t_multi:.1f} s")
print(f"speedup: {t_single / t_multi:.2f}x\n")

# ── 어드미턴스 비교 ────────────────────────────────────────────────────────────
Y_single = mf_read_admittance.read_admittance(path=results_folder, involved_blocks=["PCC-1"], file_root='single_freq')
Y_multi  = mf_read_admittance.read_admittance(path=results_folder, involved_blocks=["PCC-1"], file_root='multi_freq')

err_rel = np.abs(Y_multi.y - Y_single.y) / (np.abs(Y_single.y) + 1e-12)

print("=== 상대 오차 (Y_VSC, PCC-1) ===")
for i, j, label in [(0,0,'Ydd'), (0,1,'Ydq'), (1,0,'Yqd'), (1,1,'Yqq')]:
    e = err_rel[:, i, j]
    print(f"  {label}: mean={e.mean()*100:.4f}%  max={e.max()*100:.4f}%  "
          f"(max at {Y_single.f[e.argmax()]:.1f} Hz)")

# ── 안정도 판별 비교 ──────────────────────────────────────────────────────────
Y_grid_s = mf_read_admittance.read_admittance(path=results_folder, involved_blocks=["PCC-2"], file_root='single_freq')
Y_grid_m = mf_read_admittance.read_admittance(path=results_folder, involved_blocks=["PCC-2"], file_root='multi_freq')

L_single = np.matmul(np.linalg.inv(Y_grid_s.y), Y_single.y)
L_multi  = np.matmul(np.linalg.inv(Y_grid_m.y), Y_multi.y)

print("\n=== 안정도 판별 (GNC) ===")
stable_s = mf_stability.nyquist(L_single, Y_single.f, results_folder=results_folder, filename="gnc_single")
stable_m = mf_stability.nyquist(L_multi,  Y_multi.f,  results_folder=results_folder, filename="gnc_multi")
print(f"  single_freq: {'stable' if stable_s else 'UNSTABLE'}")
print(f"  multi_freq:  {'stable' if stable_m else 'UNSTABLE'}")
print(f"  판별 일치: {stable_s == stable_m}")
