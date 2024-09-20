from dataclasses import dataclass

import numpy as np
from loguru import logger

from pynasonde.riq.headers.default_factory import PCT_default_factory
from pynasonde.riq.headers.sct import SctType
from pynasonde.riq.utils import trim_null


@dataclass
class PctType:
    record_id: int = 0  # Sequence number of this PCT
    pri_ut: float = 0.0  # UT of this pulse
    pri_time_offset: float = 0.0  # Time read from system clock, not precise
    base_id: int = 0  # Base Frequency counter
    pulse_id: int = 0  # Pulse set element for this PRI
    ramp_id: int = 0  # Ramp set element for this PRI
    repeat_id: int = 0  # Ramp repeat element for this PRI
    loop_id: int = 0  # Outer loop element for this PRI
    frequency: float = 0.0  # Frequency of observation (kHz)
    nco_tune_word: int = 0  # Tuning word sent to the receiver
    drive_attenuation: float = 0.0  # Low-level drive attenuation [dB]
    pa_flags: int = 0  # Status flags from amplifier
    pa_forward_power: float = 0.0  # Forward power from amplifier
    pa_reflected_power: float = 0.0  # Reflected power from amplifier
    pa_vswr: float = 0.0  # Voltage Standing Wave Ratio from amplifier
    pa_temperature: float = 0.0  # Amplifier temperature
    proc_range_count: int = 0  # Number of range gates kept this PRI
    proc_noise_level: float = 0.0  # Estimated noise level for this PRI
    user: str = ""  # Spare space for user-defined information (64-character string)

    def fix_PCT_strings(self) -> None:
        logger.info("Fixing PCT strings...")
        self.user = trim_null("\x00")
        return

    def load_sct(self, sct: SctType) -> None:
        logger.info(f"Reading SCT {sct.user}")
        self.start_year, self.start_day_number, self.start_month = (
            sct.start_year,
            sct.start_daynumber,
            sct.start_month,
        )
        self.start_day, self.start_minute, self.start_second = (
            sct.start_day,
            sct.start_minute,
            sct.start_second,
        )
        self.pct_offset = sct.sounding_table_size  # bytes
        self.pct_count = sct.pulse_table_size  # bytes
        self.num_receivers = sct.receiver.rx_chan
        self.num_gates = sct.timing.gate_count
        self.echo_count = self.num_receivers * self.num_gates  # per pulse, bytes
        self.total_pulse_count = sct.timing.pri_count
        self.frequency_count = sct.frequency.base_steps
        self.max_range = sct.timing.gate_end * 0.15
        # to km
        self.min_range = sct.timing.gate_start * 0.15
        # to km
        self.tune_type = (
            sct.frequency.tune_type
        )  # 1= log 2 = linear 3 = table 4 = shuffle mode
        self.data_offset = self.pct_count + (
            4 * self.echo_count
        )  # total size of pulse table and data block in bytes
        return

    def read_pct(
        self, fname: str, pulse_num: int = 1, unicode: str = "latin-1"
    ) -> None:
        logger.info(f"Reading PCT Pulse: {pulse_num}")
        byte_offset = self.pct_offset + ((pulse_num - 1) * self.data_offset)
        # Load all PCT Type parameters
        o = np.memmap(
            fname,
            dtype=np.dtype(PCT_default_factory),
            mode="r",
            offset=byte_offset,
            shape=(1,),
        )
        for i, dtype in enumerate(PCT_default_factory):
            setattr(self, dtype[0], o[0][i])
            if (len(dtype) == 3) and (dtype[1] == "S4"):
                setattr(self, dtype[0], "".join([x.decode(unicode) for x in o[0][i]]))
        self.read_pct_IQRxRG(fname, pulse_num)
        return

    def read_pct_IQRxRG(self, fname: str, pulse_num: int = 1) -> None:
        logger.info(f"Reading IQRxRG Pulse: {pulse_num}")
        byte_offset = self.pct_offset + ((pulse_num - 1) * self.data_offset)
        factory = [("IQRxRG", "int16", (2, self.num_receivers, self.num_gates))]
        o = np.memmap(
            fname,
            dtype=np.dtype(factory),
            mode="r",
            offset=byte_offset + self.pct_count,
            shape=(1,),
        )
        # Read the data into memory for faster access
        setattr(self, "IQRxRG", o[0][0])
        # Data in the RIQ file is Big Endian
        self.IQRxRG = self.IQRxRG.byteswap()
        logger.info(f"Loaded shape of I/Q: {self.IQRxRG.shape}")
        return

    def dump_pct(self, t32: float = 0.0000186264514923096) -> None:
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
        logger.info(f"# PCT: \n{txt}")
        return
