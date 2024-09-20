import datetime as dt
from dataclasses import dataclass
from typing import List


@dataclass
class Webhook:
    URSI: str = "WI937"
    dates: List[dt.datetime] = None
    raw_url: str = "http://wallops.ionosonde.net/archive/individual/{year}/{doy}/raw/"
    raw_file_name: str = "{URSI}_{year}{doy}{hour}{min}{sec}.RIQ"
    ngi_url: str = (
        "http://wallops.ionosonde.net/archive/individual/{year}/{doy}/ionogram/"
    )
    ngi_file_name: str = "{URSI}_{year}{doy}{hour}{min}{sec}.ngi"

    def __download__(self):
        return
