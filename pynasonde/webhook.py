import datetime as dt
import os
from dataclasses import dataclass, field
from typing import List

import requests
from bs4 import BeautifulSoup
from loguru import logger
from tqdm import tqdm


@dataclass
class Webhook:
    URSI: str = "WI937"
    dates: List[dt.datetime] = field(default_factory=lambda: [dt.datetime(2024, 9, 20)])
    riq_url: str = "http://wallops.ionosonde.net/archive/individual/{year}/{doy}/raw/"
    riq_file_name: str = "{URSI}_{year}{doy}{hour}{min}{sec}.RIQ"
    ngi_url: str = (
        "http://wallops.ionosonde.net/archive/individual/{year}/{doy}/ionogram/"
    )
    ngi_file_name: str = "{URSI}_{year}{doy}{hour}{min}{sec}.ngi"

    def download(
        self, source: str = "./tmp/", itr: int = 1, ngi: bool = True, riq: bool = False
    ):
        for d in self.dates:
            self.source = os.path.join(source, d.strftime("%Y%m%d"))
            os.makedirs(self.source, exist_ok=True)
            if ngi:
                ngi_url = self.ngi_url.format(year=d.year, doy=d.timetuple().tm_yday)
                self.__dump_files__(ngi_url, self.source, ".ngi", itr)
            if riq:
                riq_url = self.riq_url.format(year=d.year, doy=d.timetuple().tm_yday)
                self.__dump_files__(riq_url, self.source, ".RIQ", itr)
        return self.source

    def __dump_files__(self, url: str, source: str, ext: str, itr: int = 1):
        logger.info(f"Checking: {url}")
        r = requests.get(url)
        I = 0
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "lxml")
            table = soup.find_all("a", href=True)
            for row in tqdm(table, total=len(table)):
                href = row.attrs["href"]
                if (self.URSI in href) and (
                    (ext.lower() in href) or (ext.upper() in href)
                ):
                    file = os.path.join(source, href)
                    if not os.path.exists(file):
                        d_url = url + href
                        logger.info(f"Downloading: {d_url}")
                        rx = requests.get(d_url)
                        if r.status_code == 200:
                            with open(file, "wb") as f:
                                f.write(rx.content)
                        else:
                            logger.info(
                                f"Error in downloading: {href} / {rx.status_code}"
                            )
                    I += 1
                    if itr == I:
                        break
        else:
            logger.info(f"Error in logging: {url} / {r.status_code}")
        return

    def __check_all_sub_folders__(
        self,
        url: str,
        base: str,
        ext: List[str],
    ):
        logger.info(f"Checking: {url}")
        r = requests.get(url)
        logger.info(f"Downloading to: {base}")
        os.makedirs(base, exist_ok=True)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "lxml")
            table = soup.find_all("a", href=True)
            for row in tqdm(table, total=len(table)):
                href = row.attrs["href"]
                if href.split(".")[-1] in ext:
                    file = os.path.join(base, href)
                    if not os.path.exists(file):
                        uri = url + href
                        logger.info(f"Downloading: {uri}")
                        rx = requests.get(uri)
                        if r.status_code == 200:
                            with open(file, "wb") as f:
                                f.write(rx.content)
                        else:
                            logger.info(
                                f"Error in downloading: {href} / {rx.status_code}"
                            )
        return
