import datetime as dt
import glob
import os
from functools import partial
from multiprocessing import Pool
from typing import List

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from pynasonde.digisonde.digi_utils import to_namespace
from pynasonde.ngi.utils import TimeZoneConversion


class EdpExtractor(object):

    def __init__(
        self,
        filename: str,
        extract_time_from_name: bool = False,
        extract_stn_from_name: bool = False,
    ):
        """
        Initialize the SkyExtractor with the given file.

        Args:
            filename (str): Path to the sky file to be processed.
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
        return

    def __update_tz__(self):
        self.stn_info = dict(
            LAT=self.edp_struct["f2"]["lat"], LONG=self.edp_struct["f2"]["lon"]
        )
        self.local_timezone_converter = TimeZoneConversion(
            lat=self.stn_info["LAT"], long=self.stn_info["LONG"]
        )
        self.local_time = self.local_timezone_converter.utc_to_local_time([self.date])[
            0
        ]
        logger.info(f"Station code: {self.stn_code}; {self.stn_info}")
        return

    def read_file(self):
        """
        Reads the file line by line into a list.

        Returns:
            list: A list of strings, each representing a line from the file.
        """
        with open(self.filename, "r") as f:
            return f.readlines()

    def __check_issues__(self, line1: str, n: int):
        """Check is there any issue with the file."""
        tag = (
            True
            if (("Problem Flags.Check" in line1) and ("failed" in line1) or (n < 5))
            else False
        )
        return tag

    def __parse_F2_datasets__(self, lines: List[str]):
        """
        Extract F2 height/density profiles
        """
        header, values = (
            list(filter(None, lines[0].replace("\n", "").split(" "))),
            list(filter(None, lines[1].replace("\n", "").split(" "))),
        )
        self.edp_struct["f2"] = dict(zip(header, values))
        for k in self.edp_struct["f2"].keys():
            self.edp_struct["f2"][k] = float(self.edp_struct["f2"][k])
        self.edp_struct["f2"]["lon"] = (self.edp_struct["f2"]["lon"] + 180) % 360 - 180
        return

    def extract(self):
        """
        Main method to extract data from the sky file and populate the sao_struct dictionary.

        Returns:
            dict: The populated sky_struct dictionary containing all extracted data.
        """
        logger.info(f"Loading file: {self.filename}")
        self.edp_struct = dict(f2=dict(), profile=[], other=dict())
        edp_arch_list = self.read_file()
        self.__parse_F2_datasets__(edp_arch_list[:2])
        self.__update_tz__()
        if not self.__check_issues__(edp_arch_list[1], len(edp_arch_list)):
            header = list(filter(None, edp_arch_list[2].replace("\n", "").split(" ")))
            head_base, head_tail = header[:7], header[7:]
            for j, l in enumerate(edp_arch_list[3:]):
                l = l[:56] + l[57:65] + l[66:]
                l = list(filter(None, l.replace("\n", "").split(" ")))
                if j == 0:
                    self.edp_struct["other"] = dict(
                        zip(head_tail, np.array(l[7:]).astype(float))
                    )
                self.edp_struct["profile"].append(
                    dict(
                        zip(
                            head_base,
                            [float(x) if i < 5 else x for i, x in enumerate(l[:7])],
                        )
                    )
                )
        else:
            logger.warning(f"Error in flags of file: {self.filename}")

        self.edp = to_namespace(self.edp_struct)
        return self.edp

    @staticmethod
    def extract_EDP(
        file: str,
        extract_time_from_name: bool = True,
        extract_stn_from_name: bool = True,
        func_name: str = "height_profile",
    ):
        def set_timestamp(o, e):
            if hasattr(e, "date"):
                o["datetime"] = e.date
            if hasattr(e, "local_time"):
                o["local_datetime"] = e.local_time
            return o

        ex = EdpExtractor(file, extract_time_from_name, extract_stn_from_name)
        ex.extract()
        if func_name == "height_profile":
            df = pd.DataFrame.from_records(ex.edp_struct["profile"])
            df = set_timestamp(df, ex)
        elif func_name == "scaled":
            df = pd.DataFrame.from_records([ex.edp_struct["f2"]])
            df = set_timestamp(df, ex)
        else:
            df = pd.DataFrame()
        return df

    @staticmethod
    def load_EDP_files(
        folders: List[str] = [],
        ext: str = "*.EDP",
        n_procs: int = 8,
        extract_time_from_name: bool = True,
        extract_stn_from_name: bool = True,
        func_name: str = "height_profile",
    ):
        collections = []
        for folder in folders:
            logger.info(f"Searching for files under: {os.path.join(folder, ext)}")
            files = glob.glob(os.path.join(folder, ext))
            files.sort()
            logger.info(f"N files: {len(files)}")
            with Pool(n_procs) as pool:
                df_collection = list(
                    tqdm(
                        pool.imap(
                            partial(
                                EdpExtractor.extract_EDP,
                                extract_time_from_name=extract_time_from_name,
                                extract_stn_from_name=extract_stn_from_name,
                                func_name=func_name,
                            ),
                            files,
                        ),
                        total=len(files),
                    )
                )
            collections.extend(df_collection)
        collections = pd.concat(collections)
        return collections


if __name__ == "__main__":
    coll = EdpExtractor.load_EDP_files(
        [
            "/media/chakras4/Crucial X9/NOAA_Archives/profilers-sounders/ionosonde/mids09/BC840/2017/233/scaled/"
        ],  # func_name="scaled"
    )
    print(coll.columns)
