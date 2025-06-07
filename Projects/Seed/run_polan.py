import datetime as dt
import glob
import os

import numpy as np
import pandas as pd

np.random.seed(1)

from pynasonde.model.polan.datasets import SimulationOutputs, Trace
from pynasonde.model.polan.polan import Polan
from pynasonde.model.polan.polan_utils import (
    get_hp_bounds_from_ht,
    get_hp_bounds_from_scale_h,
    get_Np_bounds_from_fv,
)


def get_best_guess_initial_ionosphere(
    trace: Trace,
):
    model_ionospheres = []
    for t in trace.events:
        model = "Parabolic" if t.layer == "Es" else "Chapman"
        np_bound = get_Np_bounds_from_fv(
            t.fv,
            up=0.1,
            down=0.1,
        )
        hp_bound = get_hp_bounds_from_ht(t.ht, up=20, down=20)
        hd_bound = get_hp_bounds_from_scale_h(t.ht, up=5, down=2)
        if "F1" == t.layer:
            hd_bound = [30, 60]
        if "F2" == t.layer:
            hd_bound = [40, 80]
        model_ionospheres.append(
            dict(
                model=model,
                layer=t.layer,
                np_bound=np_bound,
                hp_bound=hp_bound,
                hd_bound=hd_bound,
            )
        )
    return model_ionospheres


for d in range(1):
    date = dt.datetime(2025, 5, 19) + dt.timedelta(days=d)
    files = glob.glob(f"tmp/Digisonde/{date.strftime('%Y%m%d')}/*.XML")
    files.sort()
    records = []
    for j, f in enumerate(files):
        print(f)
        try:
            e = Trace.load_xml_sao_file(f)[0]
            fname, fig_file_name = (
                f"tmp/polan/{e.date.strftime('%Y%m%d%H%M%S')}.csv",
                f"tmp/polan/{e.date.strftime('%Y%m%d%H%M%S')}.png",
            )
            print(fname)
            if not os.path.exists(fname):
                p = Polan(
                    e, fig_file_name=fig_file_name, h_max_simulation=700, optimize=True
                )
                so = p.polan(
                    e.date,
                    model_ionospheres=get_best_guess_initial_ionosphere(e),
                    plot=True,
                    run_Es_only=True,
                    n_jobs=48,
                    optimzer_n_samples=100,
                )
                so[0].to_csv(fname)
            else:
                df = pd.read_csv(fname)
                df["date"] = pd.to_datetime(
                    fname.split("/")[-1].replace(".csv", ""), format="%Y%m%d%H%M%S"
                )
                records.append(df)
        except:
            pass
    records = pd.concat(records)
    print(records.head())
