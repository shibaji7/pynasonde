import datetime as dt
import glob

from pynasonde.model.polan.datasets import SimulationOutputs, Trace
from pynasonde.model.polan.polan import Polan
from pynasonde.model.polan.polan_utils import get_Np_bounds_from_fv


def get_best_guess_initial_ionosphere(
    trace: Trace,
    F_bounds: dict,
    E_bounds: dict,
    Es_bounds: dict,
):
    model_ionospheres = []
    for t in trace.events:
        bounds = E_bounds if t.layer in "E" else F_bounds
        if t.layer in "Es":
            bounds = Es_bounds
        model_ionospheres.append(
            dict(
                model=(
                    "Chapman" if ("F" in t.layer) or ("E" in t.layer) else "Parabolic"
                ),
                layer=t.layer,
                np_bound=bounds["np_bound"],
                hp_bound=bounds["hp_bound"],
                hd_bound=bounds["hd_bound"],
            )
        )
    return model_ionospheres


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
        # p.polan(
        #     dt.datetime(2025, 5, 27),
        #     model_ionospheres=get_best_guess_initial_ionosphere(
        #         e,
        #         F_bounds=dict(
        #             np_bound=get_Np_bounds_from_fv(
        #                 [x for ex in e.events for x in ex.fv if "F" in ex.layer], df=0.3
        #             ),
        #             hp_bound=[300, 400],
        #             hd_bound=[30, 60],
        #         ),
        #         E_bounds=dict(
        #             np_bound=get_Np_bounds_from_fv(
        #                 [x for ex in e.events for x in ex.fv if "E" in ex.layer], df=0.3
        #             ),
        #             hp_bound=[200, 300],
        #             hd_bound=[20, 40],
        #         ),
        #         Es_bounds=dict(
        #             np_bound=get_Np_bounds_from_fv(
        #                 [x for ex in e.events for x in ex.fv if "Es" in ex.description],
        #                 df=0.3,
        #             ),
        #             hp_bound=[100, 110],
        #             hd_bound=[3, 8],
        #         ),
        #     ),
        #     plot=True,
        #     run_Es_only=True,
        #     n_jobs=24,
        #     optimzer_n_samples=200,
        # )
        print(get_best_guess_initial_ionosphere(
                e,
                F_bounds=dict(
                    np_bound=get_Np_bounds_from_fv(
                        [x for ex in e.events for x in ex.fv if "F" in ex.layer], df=0.3
                    ),
                    hp_bound=[300, 400],
                    hd_bound=[30, 60],
                ),
                E_bounds=dict(
                    np_bound=get_Np_bounds_from_fv(
                        [x for ex in e.events for x in ex.fv if "E" in ex.layer], df=0.3
                    ),
                    hp_bound=[200, 300],
                    hd_bound=[20, 40],
                ),
                Es_bounds=dict(
                    np_bound=get_Np_bounds_from_fv(
                        [x for ex in e.events for x in ex.fv if "Es" in ex.description],
                        df=0.3,
                    ),
                    hp_bound=[100, 110],
                    hd_bound=[3, 8],
                ),
            ))
        break
