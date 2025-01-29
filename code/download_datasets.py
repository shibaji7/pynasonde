from pynasonde.webhook import Webhook

stn_name = "WP937"
sources = [
    dict(
        uri=f"https://data.ngdc.noaa.gov/instruments/remote-sensing/active/profilers-sounders/ionosonde/request/Eclipse/{stn_name}/2017-264/scaled/",
        folder=f"/media/chakras4/Crucial X9/NOAA_Archives/profilers-sounders/ionosonde/request/Eclipse/{stn_name}/2017/264/scaled/",
    ),
    dict(
        uri=f"https://data.ngdc.noaa.gov/instruments/remote-sensing/active/profilers-sounders/ionosonde/request/Eclipse/{stn_name}/2017-264/raw/",
        folder=f"/media/chakras4/Crucial X9/NOAA_Archives/profilers-sounders/ionosonde/request/Eclipse/{stn_name}/2017/264/raw/",
    ),
    dict(
        uri=f"https://data.ngdc.noaa.gov/instruments/remote-sensing/active/profilers-sounders/ionosonde/request/Eclipse/{stn_name}/2017-264/ionogram/",
        folder=f"/media/chakras4/Crucial X9/NOAA_Archives/profilers-sounders/ionosonde/request/Eclipse/{stn_name}/2017/264/ionogram/",
    ),
    dict(
        uri=f"https://data.ngdc.noaa.gov/instruments/remote-sensing/active/profilers-sounders/ionosonde/request/Eclipse/{stn_name}/2017-264/image/",
        folder=f"/media/chakras4/Crucial X9/NOAA_Archives/profilers-sounders/ionosonde/request/Eclipse/{stn_name}/2017/264/image/",
    ),
    dict(
        uri=f"https://data.ngdc.noaa.gov/instruments/remote-sensing/active/profilers-sounders/ionosonde/request/Eclipse/{stn_name}/2023-286/scaled/",
        folder=f"/media/chakras4/Crucial X9/NOAA_Archives/profilers-sounders/ionosonde/request/Eclipse/{stn_name}/2023/286/scaled/",
    ),
    dict(
        uri=f"https://data.ngdc.noaa.gov/instruments/remote-sensing/active/profilers-sounders/ionosonde/request/Eclipse/{stn_name}/2023-286/raw/",
        folder=f"/media/chakras4/Crucial X9/NOAA_Archives/profilers-sounders/ionosonde/request/Eclipse/{stn_name}/2023/286/raw/",
    ),
    dict(
        uri=f"https://data.ngdc.noaa.gov/instruments/remote-sensing/active/profilers-sounders/ionosonde/request/Eclipse/{stn_name}/2023-286/ionogram/",
        folder=f"/media/chakras4/Crucial X9/NOAA_Archives/profilers-sounders/ionosonde/request/Eclipse/{stn_name}/2023/286/ionogram/",
    ),
    dict(
        uri=f"https://data.ngdc.noaa.gov/instruments/remote-sensing/active/profilers-sounders/ionosonde/request/Eclipse/{stn_name}/2023-286/image/",
        folder=f"/media/chakras4/Crucial X9/NOAA_Archives/profilers-sounders/ionosonde/request/Eclipse/{stn_name}/2023/286/image/",
    ),
    dict(
        uri=f"https://data.ngdc.noaa.gov/instruments/remote-sensing/active/profilers-sounders/ionosonde/request/Eclipse/{stn_name}/098/scaled/",
        folder=f"/media/chakras4/Crucial X9/NOAA_Archives/profilers-sounders/ionosonde/request/Eclipse/{stn_name}/2024/098/scaled/",
    ),
    dict(
        uri=f"https://data.ngdc.noaa.gov/instruments/remote-sensing/active/profilers-sounders/ionosonde/request/Eclipse/{stn_name}/098/raw/",
        folder=f"/media/chakras4/Crucial X9/NOAA_Archives/profilers-sounders/ionosonde/request/Eclipse/{stn_name}/2024/098/raw/",
    ),
    dict(
        uri=f"https://data.ngdc.noaa.gov/instruments/remote-sensing/active/profilers-sounders/ionosonde/request/Eclipse/{stn_name}/098/ionogram/",
        folder=f"/media/chakras4/Crucial X9/NOAA_Archives/profilers-sounders/ionosonde/request/Eclipse/{stn_name}/2024/098/ionogram/",
    ),
    dict(
        uri=f"https://data.ngdc.noaa.gov/instruments/remote-sensing/active/profilers-sounders/ionosonde/request/Eclipse/{stn_name}/098/image/",
        folder=f"/media/chakras4/Crucial X9/NOAA_Archives/profilers-sounders/ionosonde/request/Eclipse/{stn_name}/2024/098/image/",
    ),
]
wh = Webhook()
for source in sources:
    wh.__check_all_sub_folders__(
        source["uri"],
        source["folder"],
        ["SAO", "EDP", "PNG", "MMM", "16C"],
    )
