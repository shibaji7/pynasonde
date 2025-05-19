import datetime as dt
import shutil

from pynasonde.ngi.source import DataSource


def generate_fti_profiles(folder, fig_file_name, fig_title="", stn="", flim=[3.5, 4.5]):
    import os

    os.makedirs(f"tmp/fti/{flim[0]}", exist_ok=True)
    ds = DataSource(source_folder=folder)
    ds.load_data_sets(0, -1)
    ds.extract_FTI_RTI(folder=f"tmp/fti/{flim[0]}", rlim=[50, 400], flim=flim)
    return


## Analyzing the dataset form Speed Deamon 2022
for doy in range(233, 234, 1):
    stn = "WI937"
    date = dt.datetime(2022, 1, 1) + dt.timedelta(days=doy - 1)
    fig_file_name = f"../../tmp/FTI.{stn}.2022.doy-{doy}.png"
    fig_title = f"Speed Demon / {date.strftime('%Y-%m-%d')}"

    shutil.rmtree(f"/tmp/{doy}/", ignore_errors=True)
    shutil.copytree(
        f"/media/chakras4/ERAU/SpeedDemon/WI937/individual/2022/{doy}/ionogram/",
        f"/tmp/{doy}/ionogram/",
    )
    generate_fti_profiles(
        folder=f"/tmp/{doy}/ionogram/",
        fig_file_name=fig_file_name,
        fig_title=fig_title,
        stn=stn,
    )
    shutil.rmtree(f"/tmp/{doy}/", ignore_errors=True)
