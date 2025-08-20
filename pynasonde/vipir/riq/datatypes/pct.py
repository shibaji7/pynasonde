import struct
from dataclasses import dataclass

import numpy as np
from loguru import logger

from pynasonde.vipir.riq.datatypes.default_factory import PCT_default_factory
from pynasonde.vipir.riq.datatypes.sct import SctType, read_dtype
from pynasonde.vipir.riq.utils import trim_null


@dataclass
class PctType:
    record_id: np.int32 = 0  # Sequence number of this PCT
    pri_ut: np.float64 = 0.0  # UT of this pulse
    pri_time_offset: np.float64 = 0.0  # Time read from system clock, not precise
    base_id: np.int32 = 0  # Base Frequency counter
    pulse_id: np.int32 = 0  # Pulse set element for this PRI
    ramp_id: np.int32 = 0  # Ramp set element for this PRI
    repeat_id: np.int32 = 0  # Ramp repeat element for this PRI
    loop_id: np.int32 = 0  # Outer loop element for this PRI
    frequency: np.float64 = 0.0  # Frequency of observation (kHz)
    nco_tune_word: np.int32 = 0  # Tuning word sent to the receiver
    drive_attenuation: np.float64 = 0.0  # Low-level drive attenuation [dB]
    pa_flags: np.int32 = 0  # Status flags from amplifier
    pa_forward_power: np.float64 = 0.0  # Forward power from amplifier
    pa_reflected_power: np.float64 = 0.0  # Reflected power from amplifier
    pa_vswr: np.float64 = 0.0  # Voltage Standing Wave Ratio from amplifier
    pa_temperature: np.float64 = 0.0  # Amplifier temperature
    proc_range_count: np.int32 = 0  # Number of range gates kept this PRI
    proc_noise_level: np.float64 = 0.0  # Estimated noise level for this PRI
    user: str = ""  # Spare space for user-defined information (64-character string)

    def fix_PCT_strings(self) -> None:
        logger.info("Fixing PCT strings...")
        self.user = trim_null("\x00")
        return

    def read_pct_from_file_pointer(
        self, fp, sct: SctType, vipir_version: dict, unicode: str = "latin-1"
    ):
        for i, dtype in enumerate(PCT_default_factory):
            self = read_dtype(dtype, self, fp, unicode)
        logger.info(f"Reading PCT {self.record_id}")
        self.load_sct(sct, vipir_version)
        vipir_value_size = (vipir_version["vipir_version"] + 1) * 2
        chunksize = (
            int(vipir_value_size) * 2 * sct.station.rx_count * sct.timing.gate_count
        )
        data = fp.read(chunksize)
        self.pulse_i = np.full(
            (sct.timing.gate_count, sct.station.rx_count),
            32767,
            dtype=vipir_version["np_format"],
        )
        self.pulse_q = np.full(
            (sct.timing.gate_count, sct.station.rx_count),
            32767,
            dtype=vipir_version["np_format"],
        )
        index, index_increment = 0, 2 * vipir_value_size
        for j in range(sct.timing.gate_count):
            for k in range(sct.station.rx_count):
                self.pulse_i[j, k], self.pulse_q[j, k] = (
                    struct.unpack("<i", data[index : index + vipir_value_size])[0],
                    struct.unpack(
                        "<i", data[index + vipir_value_size : index + index_increment]
                    )[0],
                )
                index += index_increment
        return self

    def load_sct(self, sct: SctType, vipir_version: dict) -> None:
        logger.info(f"Reading SCT {sct.user}")
        self.vipir_version = vipir_version
        self.sct_offset = sct.sounding_table_size  # bytes
        self.pct_offset = sct.pulse_table_size  # bytes
        self.num_receivers = sct.receiver.rx_chan
        self.num_gates = sct.timing.gate_count
        self.echo_count = (
            sct.receiver.rx_chan * sct.timing.gate_count
        )  # per pulse, bytes
        self.total_pulse_count = (
            sct.timing.pri_count
        )  # Total pulse count for integration
        # total size of pulse table and data block in bytes
        # IQ ~ 2, Value Size (2 or 4 or 8), n_echoes
        self.data_offset = self.pct_offset + (
            2 * (vipir_version["vipir_version"] + 1) * self.echo_count
        )
        return

    def dump_pct(
        self, t32: np.float64 = 0.0000186264514923096, to_file: str = None
    ) -> None:
        self.fix_PCT_strings()
        txt = f"{'pct.record_id':<30}{self.record_id:>12}\n"
        txt += f"{'pct.pri_ut':<30}{self.pri_ut:>12.2f}\n"
        txt += f"{'pct.pri_time_offset':<30}{self.pri_time_offset:>12.2f}\n"
        txt += f"{'pct.base_id':<30}{self.base_id:>12}\n"
        txt += f"{'pct.pulse_id':<30}{self.pulse_id:>12}\n"
        txt += f"{'pct.ramp_id':<30}{self.ramp_id:>12}\n"
        txt += f"{'pct.repeat_id':<30}{self.repeat_id:>12}\n"
        txt += f"{'pct.frequency':<30}{self.frequency:>12.2f} {self.frequency:>12.6e}\n"
        txt += f"{'pct.nco_tune_word':<30}0x{self.nco_tune_word:08X} {self.nco_tune_word:>12} {t32 * float(self.nco_tune_word):12.3f}\n"
        txt += f"{'pct.drive_attenuation':<30}{self.drive_attenuation:>12.2f}\n"
        txt += f"{'pct.pa_flags':<30}0x{self.pa_flags:08X}\n"
        txt += f"{'pct.pa_forward_power':<30}{self.pa_forward_power:>12.2f}\n"
        txt += f"{'pct.pa_reflected_power':<30}{self.pa_reflected_power:>12.2f}\n"
        txt += f"{'pct.pa_vswr':<30}{self.pa_vswr:>12.2f}\n"
        txt += f"{'pct.pa_temperature':<30}{self.pa_temperature:>12.2f}\n"
        txt += f"{'pct.procq_range_count':<30}{self.proc_range_count:>12}\n"
        txt += f"{'pct.proc_noise_level':<30}{self.proc_noise_level:>12.2f}\n"
        txt += f"{'pct.user:':<30}{self.user.strip()}\n"

        if to_file:
            with open(to_file, "w") as f:
                f.write(txt)
        else:
            logger.info(f"# PCT: \n {txt}")
        return
