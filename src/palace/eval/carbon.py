"""CodeCarbon wrapper (offline). On a laptop, power may be ESTIMATED (RAPL/GPU via NVML or TDP fallback): the mode is recorded; treat as estimate."""
from __future__ import annotations
import time, warnings, logging
from contextlib import contextmanager

@contextmanager
def track(name="run", enabled=True):
    info = {"kwh": float("nan"), "kg_co2e": float("nan"), "mode": "disabled", "wall_s": 0.0}; t0 = time.time(); tr = None
    if enabled:
        try:
            from codecarbon import OfflineEmissionsTracker
            logging.getLogger("codecarbon").setLevel(logging.ERROR)
            tr = OfflineEmissionsTracker(project_name=name, country_iso_code="IND", save_to_file=False, log_level="error", measure_power_secs=10, allow_multiple_runs=True)
            tr.start()
        except Exception as e:
            info["mode"] = f"codecarbon_failed:{type(e).__name__}"; tr = None
    try: yield info
    finally:
        info["wall_s"] = time.time() - t0
        if tr is not None:
            try:
                tr.stop(); d = tr.final_emissions_data
                info.update(kwh=float(d.energy_consumed), kg_co2e=float(d.emissions), mode=f"codecarbon_offline(cpu={getattr(d,'cpu_model','?')};gpu={getattr(d,'gpu_model','?')};tracking={getattr(tr,'_tracking_mode','?')})",
                            emission_factor_kg_per_kwh=float(d.emissions / d.energy_consumed) if d.energy_consumed else float("nan"))
            except Exception as e: info["mode"] = f"codecarbon_stop_failed:{type(e).__name__}"
