"""
Compare single-frequency vs multi-frequency scan: time and admittance accuracy

[주의] multi_freq_scan은 passive/network scan (topology 인자 필요) 에서만 동작합니다.
       topology=None 인 single-bus active scan 경로(line 687, site-packages)는
       multi_freq_scan 분기가 없으므로 이 스크립트는 active scan 결과 재현성 확인
       및 경로/파라미터 검증 용도로만 활용하십시오.
       multi_freq_scan 효과를 측정하려면 topology 파일이 있는 multi-bus 모델이 필요합니다.
"""
import os
import time
import numpy as np
from ztoolacdc import *

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

freq = create_freq.loglist(f_min=f_min, f_max=f_max, f_points=f_points, f_base=f_base)

common = dict(
    t_snap=t_snap, t_sim=t_sim, t_step=t_step, dt_injections=dt_injections,
    f_base=f_base, freq=freq, start_fft=start_fft, fft_periods=fft_periods,
    v_perturb_mag=v_perturb_mag, working_dir=pscad_folder,
    workspace_name=workspace_name, project_name=project_name,
    results_folder=results_folder,
)

# ── Run 1: single frequency ──────────────────────────────────────────────────
t0 = time.time()
frequency_sweep.frequency_sweep(**common, output_files='single_freq', multi_freq_scan=False)
t_single = time.time() - t0
print(f"\n[single_freq] elapsed: {t_single:.1f} s")

# ── Run 2: multi frequency (topology=None 이므로 active scan 경로 → 실질적으로 동일) ─
t0 = time.time()
frequency_sweep.frequency_sweep(**common, output_files='multi_freq', multi_freq_scan=True)
t_multi = time.time() - t0
print(f"[multi_freq]  elapsed: {t_multi:.1f} s")
print(f"speedup: {t_single / t_multi:.2f}x  (active scan이면 차이 없음)\n")

# ── 어드미턴스 비교 ────────────────────────────────────────────────────────────
Y_single = read_admittance.read_admittance(path=results_folder, involved_blocks=["PCC-1"], file_root='single_freq')
Y_multi  = read_admittance.read_admittance(path=results_folder, involved_blocks=["PCC-1"], file_root='multi_freq')

err_rel = np.abs(Y_multi.y - Y_single.y) / (np.abs(Y_single.y) + 1e-12)

print("=== 상대 오차 (Y_VSC, PCC-1) ===")
for i, j, label in [(0,0,'Ydd'), (0,1,'Ydq'), (1,0,'Yqd'), (1,1,'Yqq')]:
    e = err_rel[:, i, j]
    print(f"  {label}: mean={e.mean()*100:.4f}%  max={e.max()*100:.4f}%  "
          f"(max at {Y_single.f[e.argmax()]:.1f} Hz)")

# ── 안정도 판별 비교 ──────────────────────────────────────────────────────────
Y_grid_s = read_admittance.read_admittance(path=results_folder, involved_blocks=["PCC-2"], file_root='single_freq')
Y_grid_m = read_admittance.read_admittance(path=results_folder, involved_blocks=["PCC-2"], file_root='multi_freq')

L_single = np.matmul(np.linalg.inv(Y_grid_s.y), Y_single.y)
L_multi  = np.matmul(np.linalg.inv(Y_grid_m.y), Y_multi.y)

print("\n=== 안정도 판별 (GNC) ===")
stable_s = stability.nyquist(L_single, Y_single.f, results_folder=results_folder, filename="gnc_single")
stable_m = stability.nyquist(L_multi,  Y_multi.f,  results_folder=results_folder, filename="gnc_multi")
print(f"  single_freq: {'stable' if stable_s else 'UNSTABLE'}")
print(f"  multi_freq:  {'stable' if stable_m else 'UNSTABLE'}")
print(f"  판별 일치: {stable_s == stable_m}")
