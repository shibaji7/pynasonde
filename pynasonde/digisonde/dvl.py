import datetime as dt
import glob
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from pynasonde.digisonde.digi_utils import to_namespace


class DvlExtractor(object):

    def __init__(
        self,
        filename: str,
        extract_time_from_name: bool = False,
        extract_stn_from_name: bool = False,
    ):
        """
        Initialize the DVLExtractor with the given file.

        Args:
            filename (str): Path to the DVL file to be processed.
        """
        # Initialize the data structure to hold extracted data
        self.filename = filename
        if extract_time_from_name:
            date = self.filename.split("_")[-1].replace(".SAO", "").replace(".sao", "")
            self.date = dt.datetime(int(date[:4]), 1, 1) + dt.timedelta(
                int(date[4:7]) - 1
            )
            self.date = self.date.replace(
                hour=int(date[7:9]), minute=int(date[9:11]), second=int(date[11:13])
            )
            logger.info(f"Date: {self.date}")
        if extract_stn_from_name:
            self.stn_code = self.filename.split("/")[-1].split("_")[0]
            logger.info(f"Station code: {self.stn_code}")
        self.key_order = [
            "type",
            "version",
            "station_id",
            "ursi_tag",
            "lat",
            "lon",
            "date",
            "doy",
            "time",
            "Vx",
            "Vx_err",
            "Vy",
            "Vy_err",
            "Az",
            "Az_err",
            "Vh",
            "Vh_err",
            "Vz",
            "Vz_err",
            "Cord",
            "Hb",
            "Ht",
            "Fl",
            "Fu",
        ]
        self.dvl_struct = dict(
            type="",
            version="",
            station_id=np.nan,
            ursi_tag="",
            lat=np.nan,
            lon=np.nan,
            date=self.date.date(),
            doy=self.date.timetuple().tm_yday,
            time=self.date.time(),
            Vx=np.nan,  # in m/s; magnetic north direction
            Vx_err=np.nan,
            Vy=np.nan,  # in m/s; magnetic east direction
            Vy_err=np.nan,
            Az=np.nan,  # in m/s; counted clockwise from the magnetic north
            Az_err=np.nan,
            Vh=np.nan,
            Vh_err=np.nan,
            Vz=np.nan,  # in m/s; vertical velocity component
            Vz_err=np.nan,  #
            Cord="",  # in COM means Compass, GEO means Geographic, CGm means Corrected Geromagnetic
            Hb=np.nan,  # in km
            Ht=np.nan,  # in km
            Fl=np.nan,  # in MHz
            Fu=np.nan,  # in MHz
        )

        return

    def read_file(self):
        """
        Reads the file line by line into a list.

        Returns:
            list: A list of strings, each representing a line from the file.
        """
        with open(self.filename, "r") as f:
            return f.readlines()

    def extract(self):
        """
        Main method to extract data from the DVL file and populate the sao_struct dictionary.

        Returns:
            dict: The populated dvl_struct dictionary containing all extracted data.
        """
        # Read file lines
        dvl_arch = self.read_file()
        dvl_arch = list(filter(None, dvl_arch[0].split()))
        for i, key in enumerate(self.key_order):
            if type(self.dvl_struct[key]) == dt.date:
                logger.info(
                    f"Already holds date information: {self.dvl_struct['date']}"
                )
            elif type(self.dvl_struct[key]) == dt.time:
                logger.info(
                    f"Already holds time information: {self.dvl_struct['time']}"
                )
            else:
                self.dvl_struct[key] = type(self.dvl_struct[key])(dvl_arch[i])
        self.dvl = to_namespace(self.dvl_struct)
        return self.dvl_struct

    @staticmethod
    def extract_DVL_pandas(
        file: str,
        extract_time_from_name: bool = True,
        extract_stn_from_name: bool = True,
    ):
        extractor = DvlExtractor(file, extract_time_from_name, extract_stn_from_name)
        df = pd.DataFrame.from_records([extractor.extract()])
        df["datetime"] = extractor.date
        return df

    @staticmethod
    def load_DVL_files(
        folders: str = "tmp/SKYWAVE_DPS4D_2023_10_13",
        ext: str = "*.DVL",
        n_procs: int = 4,
        extract_time_from_name: bool = True,
        extract_stn_from_name: bool = True,
    ):
        logger.info(f"Searching for files under: {os.path.join(folders, ext)}")
        files = glob.glob(os.path.join(folders, ext))
        files.sort()
        logger.info(f"N files: {len(files)}")
        with Pool(n_procs) as pool:
            df_collection = list(
                tqdm(
                    pool.imap(
                        partial(
                            DvlExtractor.extract_DVL_pandas,
                            extract_time_from_name=extract_time_from_name,
                            extract_stn_from_name=extract_stn_from_name,
                        ),
                        files,
                    ),
                    total=len(files),
                )
            )
        df_collection = pd.concat(df_collection)
        return df_collection


if __name__ == "__main__":
    # extractor = DvlExtractor("tmp/KR835_2023286235715.DVL", True, True)
    # extractor.extract()
    collection = DvlExtractor.load_DVL_files()
    print(collection.columns)
    from pynasonde.digisonde.digi_plots import SkySummaryPlots

    SkySummaryPlots.plot_dvl_drift_velocities(collection, fname="tmp/extract_dvl.png")
    # sky = SkySummaryPlots(figsize=(8, 4), subplot_kw=None)
    # sky.plot_dvl_drift_velocities(collection)
    # sky.save("tmp/extract_dvl.png")
    # sky.close()
