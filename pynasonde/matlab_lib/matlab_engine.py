import matlab
import matlab.engine
import numpy as np
import pandas as pd
from loguru import logger


def get_matlab_pynasonde_lib():
    import importlib.util
    import os
    import pathlib

    pynasonde_spec = importlib.util.find_spec("pynasonde")
    if pynasonde_spec is None:
        raise ImportError("pynasonde module not found")
    pynasonde_path = pathlib.Path(pynasonde_spec.origin).parent
    lib_path = pynasonde_path / "matlab_lib"
    if lib_path.exists():
        logger.info(f"Matlab library path found: {lib_path}")
        return str(lib_path)
    else:
        raise FileNotFoundError(f"Matlab library path not found: {lib_path}")


class CreateFig:

    def __init__(self, fig_path: str = "figures/", lib_path: str = None):
        self.fig_path = fig_path
        self.eng = matlab.engine.start_matlab()
        env_path = get_matlab_pynasonde_lib() if lib_path is None else lib_path
        self.eng.addpath(self.eng.genpath(env_path), nargout=0)
        logger.info("Matlab engine started and library path added.")
        return

    def close(self):
        logger.info("Closing Matlab engine.")
        self.eng.quit()
        return

    def generate_scaled_TS_figure(self, data_dicts: list[dict], fig_file_name: str):
        self.eng.eval(
            f"sp = SaoSummaryPlots('', {len(data_dicts)},1,16, [8 5*{len(data_dicts)}]);",
            nargout=0,
        )
        for i, data_dict in enumerate(data_dicts):
            (
                df,
                datetime_key,
                xlim,
                right_yparams,
                left_yparams,
                left_param_labels,
                right_param_labels,
                color_direction,
                title_txt,
                vlines,
                vline_styles,
            ) = (
                data_dict["dataset"],
                data_dict.get("datetime_key", "datetime"),
                data_dict.get("xlim", []),
                data_dict.get("right_yparams", ["hmF2", "hmE"]),
                data_dict.get("left_yparams", ["foF2", "foE"]),
                data_dict.get("left_param_labels", ["hmF_2", "hmE"]),
                data_dict.get("right_param_labels", ["foF_2", "foE"]),
                data_dict.get("color_direction", "dark2light"),
                data_dict.get("title_txt", f"({chr(65+i)})"),
                data_dict.get("vlines", []),
                data_dict.get("vline_styles", []),
            )

            num_cols, str_cols = (
                df.select_dtypes(include=["number"]).columns,
                df.columns.difference(df.select_dtypes(include=["number"]).columns),
            )

            # numeric as matlab.double
            num_mat, dt_str, str_mat = (
                matlab.double(df[num_cols].to_numpy(dtype=float).tolist()),
                df[datetime_key]
                .dt.strftime("%Y-%m-%d %H:%M:%S.%f")
                .where(df[datetime_key].notna(), ""),
                [
                    [
                        s if pd.notna(s) else ""
                        for s in df[str_cols].iloc[i].astype(str).tolist()
                    ]
                    for i in range(len(df))
                ],
            )

            self.eng.workspace["strMat"] = str_mat
            self.eng.workspace["numMat"] = num_mat
            self.eng.workspace["dtStr"] = dt_str.tolist()
            self.eng.workspace["numCols"] = list(num_cols)
            self.eng.workspace["strCols"] = list(str_cols)
            self.eng.workspace["xlimStr"] = [
                x.strftime("%Y-%m-%d %H:%M:%S") for x in xlim
            ]
            self.eng.workspace["left_yparams"] = list(left_yparams)
            self.eng.workspace["right_yparams"] = list(right_yparams)
            self.eng.workspace["left_yparam_labels"] = list(left_param_labels)
            self.eng.workspace["right_yparam_labels"] = list(right_param_labels)
            self.eng.workspace["title_txt"] = title_txt
            self.eng.workspace["vlines"] = matlab.double(vlines)
            self.eng.workspace["vline_styles"] = list(vline_styles)

            self.eng.eval(
                f"""
                    xlim = datetime(xlimStr,"InputFormat","yyyy-MM-dd HH:mm:ss");
                    T = array2table(numMat, "VariableNames", numCols);
                    T.datetime = datetime(dtStr(:), "InputFormat","yyyy-MM-dd HH:mm:ss.SSSSSS");
                    [ax, tax] = sp.plot_TS(...
                            T, "datetime", left_yparams, right_yparams, ...
                            xlim=xlim, left_yparam_labels=left_yparam_labels, ...
                            right_yparam_labels=right_yparam_labels, ...
                            color_direction = "dark2light", ms=3, ...
                            title_txt=title_txt, txt_pos=[0.9 0.9] ...
                    );
                    sp.save(fullfile("{self.fig_path}", "{fig_file_name}"));
                    sp.close();
                """,
                nargout=0,
            )
        return
