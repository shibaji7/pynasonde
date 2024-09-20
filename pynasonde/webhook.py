import datetime as dt
import os
from dataclasses import dataclass, field
from typing import List

import requests
from bs4 import BeautifulSoup
from loguru import logger


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

    def download(self, source: str = "./tmp/"):
        for d in self.dates:
            source = os.path.join(source, d.strftime("%Y%m%d"))
            os.makedirs(source, exist_ok=True)
            ngi_url = self.ngi_url.format(year=d.year, doy=d.timetuple().tm_yday)
            self.__dump_files__(ngi_url, source, ".ngi")
            riq_url = self.riq_url.format(year=d.year, doy=d.timetuple().tm_yday)
            self.__dump_files__(riq_url, source, ".RIQ")
        return

    def __dump_files__(self, url: str, source: str, ext: str):
        logger.info(f"Checking: {url}")
        r = requests.get(url)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "lxml")
            table = soup.find_all("a", href=True)
            for row in table:
                href = row.attrs["href"]
                if (self.URSI in href) and (
                    (ext.lower() in href) or (ext.upper() in href)
                ):
                    url = url + href
                    rx = requests.get(url)
                    if r.status_code == 200:
                        file = os.path.join(source, href)
                        with open(file, "wb") as f:
                            f.write(rx.content)
                    else:
                        logger.info(f"Error in downloading: {href} / {rx.status_code}")
                    break
        else:
            logger.info(f"Error in logging: {url} / {r.status_code}")
        return
