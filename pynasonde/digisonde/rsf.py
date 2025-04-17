import datetime as dt

from loguru import logger

from pynasonde.digisonde.digi_utils import get_digisonde_info
from pynasonde.ngi.utils import TimeZoneConversion


class RsfExtractor(object):

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
            self.stn_info = get_digisonde_info(self.stn_code)
            self.local_timezone_converter = TimeZoneConversion(
                lat=self.stn_info["LAT"], long=self.stn_info["LONG"]
            )
            self.local_time = self.local_timezone_converter.utc_to_local_time(
                [self.date]
            )[0]
            logger.info(f"Station code: {self.stn_code}; {self.stn_info}")
        return

    def extract(self):
        """
        Main method to extract data from the rsf file and populate the rsf_struct dictionary.

        Returns:
            dict: The populated rsf_struct dictionary containing all extracted data.
        """
        return


if __name__ == "__main__":
    extractor = RsfExtractor(
        "tmp/SKYWAVE_DPS4D_2023_10_13/KR835_2023286235456.RSF", True, True
    )
    extractor.extract()
