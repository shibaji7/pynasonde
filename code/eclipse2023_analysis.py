from pynasonde.digisonde.dvl import DvlExtractor
from pynasonde.digisonde.sao import SaoExtractor

sao_dataset = SaoExtractor.load_SAO_files(folders=[], func_name="height_profile")
drift_dataset = DvlExtractor.load_DVL_files(folders=[])
