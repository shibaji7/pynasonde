::: pynasonde.digisonde.digi_utils
    handler: python
    options:
        show_root_heading: true
        show_source: false
        members:
            - to_namespace
            - setsize
            - get_gridded_parameters
            - load_station_csv
            - load_dtd_file
            - is_valid_xml_data_string


## Examples

```py
from pynasonde.digisonde.digi_utils import load_station_csv, get_digisonde_info

stations = load_station_csv()
info = get_digisonde_info('KR835')
print(info)
```
