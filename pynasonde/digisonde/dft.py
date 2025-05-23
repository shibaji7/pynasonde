import datetime as dt
import struct

import numpy as np
from loguru import logger


class DftExtractor(object):

    def __init__(
        self,
        filename: str,
        extract_time_from_name: bool = False,
        extract_stn_from_name: bool = False,
        DATA_BLOCK_SIZE: int = 4096,
    ):
        """
        Initialize the DftExtractor with the given file.

        Args:
            filename (str): Path to the dft file to be processed.
        """
        # Initialize the data structure to hold extracted data
        self.filename = filename
        self.DATA_BLOCK_SIZE = DATA_BLOCK_SIZE
        with open(self.filename, "rb") as file:
            self.BLOCKS = int(len(file.read()) / self.DATA_BLOCK_SIZE)
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
        return

    def extract(self):
        """
        Main method to extract data from the dft file and populate the sao_struct dictionary.

        Returns:
            dict: The populated dft_struct dictionary containing all extracted data.
        """
        with open(self.filename, "rb") as file:
            for n in range(self.BLOCKS):
                logger.debug(f"Reading block {n+1} of {self.BLOCKS}")
                print(struct.unpack("B", file.read(1))[0])
                break
        return


if __name__ == "__main__":
    extractor = DftExtractor(
        "tmp/SKYWAVE_DPS4D_2023_10_13/KR835_2023286235715.DFT", True, True
    )
    extractor.extract()
