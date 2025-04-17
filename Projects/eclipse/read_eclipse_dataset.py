import sys

from pynasonde.digisonde.dvl import DvlExtractor
from pynasonde.digisonde.sao import SaoExtractor

sys.path.append("Projects/eclipse/")
import utils


class Digisonde(self):

    def __init__(self, folders):
        self.folders = folders
        return

    def read_sao(self):
        self.sao_pf_df = SaoExtractor.load_SAO_files(
            folders=self.folders,
            func_name="height_profile",
            n_procs=12,
        )
        self.sao_sc_df = SaoExtractor.load_SAO_files(
            folders=self.folders,
            func_name="scaled",
            n_procs=12,
        )
        return

    def read_dvl(self):
        self.dvl_df = DvlExtractor.load_DVL_files(
            self.folders,
            n_procs=12,
        )
        return

    def read_sky(self):
        # self.sky_df = DvlExtractor.load_DVL_files(
        #     self.folders,
        #     n_procs=12,
        # )
        return

    def load_eclipse(self, time, lat, lon):
        self.obs = utils.create_eclipse_path_local(time, lat, lon)
        return
