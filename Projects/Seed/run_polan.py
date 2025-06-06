import datetime as dt
import glob

from pynasonde.model.polan.datasets import SimulationOutputs, Trace
from pynasonde.model.polan.polan import Polan
from pynasonde.model.polan.polan_utils import get_Np_bounds_from_fv

for d in range(1):
    date = dt.datetime(2025, 5, 19) + dt.timedelta(days=d)
    files = glob.glob(f"tmp/Digisonde/{date.strftime('%Y%m%d')}/*.XML")
    files.sort()
    for f in files:
        print(f)
        e = Trace.load_xml_sao_file(f)[0]
        p = Polan(
            e, fig_file_name="tmp/polan/sample.png", h_max_simulation=700, optimize=True
        )
        p.polan(
            dt.datetime(2025, 5, 27),
            model_ionospheres=[
                dict(
                    model="Chapman",
                    layer="F",
                    np_bound=get_Np_bounds_from_fv(
                        [x for ex in e.events for x in ex.fv], df=0.3
                    ),
                    hp_bound=[300, 400],
                    hd_bound=[30, 60],
                ),
                dict(
                    model="Parabolic",
                    layer="Es",
                    np_bound=get_Np_bounds_from_fv(
                        [x for ex in e.events for x in ex.fv if "Es" in ex.description],
                        df=0.3,
                    ),
                    hp_bound=[100, 110],
                    hd_bound=[3, 8],
                ),
            ],
            plot=True,
            run_Es_only=True,
            n_jobs=24,
            optimzer_n_samples=200,
        )
        break
