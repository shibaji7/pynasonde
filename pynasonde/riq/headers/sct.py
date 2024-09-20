from cmath import rect  # For complex numbers
from dataclasses import dataclass, field
from typing import List

import numpy as np
from loguru import logger

from pynasonde.riq.headers.default_factory import (
    Exciter_default_factory,
    Frequency_default_factory,
    Monitor_default_factory,
    Reciever_default_factory,
    SCT_default_factory,
    Station_default_factory,
    Timing_default_factory,
)
from pynasonde.riq.utils import trim_null


@dataclass
class StationType:
    file_id: str = ""
    ursi_id: str = ""
    rx_name: str = ""
    rx_latitude: float = 0.0
    rx_longitude: float = 0.0
    rx_altitude: float = 0.0
    rx_count: int = 0
    rx_antenna_type: str = ""
    rx_position: List[List[float]] = field(
        default_factory=lambda: [[0.0] * 32 for _ in range(3)]
    )
    rx_direction: List[List[float]] = field(
        default_factory=lambda: [[0.0] * 32 for _ in range(3)]
    )
    rx_height: List[float] = field(default_factory=lambda: [0.0] * 32)
    rx_cable_length: List[float] = field(default_factory=lambda: [0.0] * 32)
    frontend_atten: float = 0.0
    tx_name: str = ""
    tx_latitude: float = 0.0
    tx_longitude: float = 0.0
    tx_altitude: float = 0.0
    tx_antenna_type: str = ""
    tx_vector: List[float] = field(default_factory=lambda: [0.0] * 3)
    tx_height: float = 0.0
    tx_cable_length: float = 0.0
    drive_band_count: int = 0
    drive_band_bounds: List[List[float]] = field(
        default_factory=lambda: [[0.0] * 64 for _ in range(2)]
    )
    drive_band_atten: List[float] = field(default_factory=lambda: [0.0] * 64)
    rf_control: int = -1
    ref_type: str = ""
    clock_type: str = ""
    user: str = ""

    def read_station(self, fname: str, unicode: str = "latin-1") -> None:
        # Load all Station Type parameters
        o = np.memmap(
            fname,
            dtype=np.dtype(Station_default_factory),
            mode="r",
            offset=316,
            shape=(1,),
        )
        for i, dtype in enumerate(Station_default_factory):
            setattr(self, dtype[0], o[0][i])
            if (len(dtype) == 3) and (dtype[1] == "S4"):
                setattr(self, dtype[0], "".join([x.decode(unicode) for x in o[0][i]]))
        return


@dataclass
class TimingType:
    file_id: str = ""
    pri: float = 0.0
    pri_count: int = 0
    ionogram_count: int = 0
    holdoff: float = 0.0
    range_gate_offset: float = 0.0
    gate_count: int = 0
    gate_start: float = 0.0
    gate_end: float = 0.0
    gate_step: float = 0.0
    data_start: float = 0.0
    data_width: float = 0.0
    data_baud_count: int = 0
    data_wave_file: str = ""
    data_baud: List[complex] = field(default_factory=lambda: [complex(0.0, 0.0)] * 1024)
    data_pairs: int = 0
    cal_start: float = 0.0
    cal_width: float = 0.0
    cal_baud_count: int = 0
    cal_wave_file: str = ""
    cal_baud: List[complex] = field(default_factory=lambda: [complex(0.0, 0.0)] * 1024)
    cal_pairs: int = 0
    user: str = ""

    def read_timing(self, fname: str, unicode: str = "latin-1") -> None:
        # Load all Timing Type parameters
        o = np.memmap(
            fname,
            dtype=np.dtype(Timing_default_factory),
            mode="r",
            offset=3552,
            shape=(1,),
        )
        for i, dtype in enumerate(Timing_default_factory):
            setattr(self, dtype[0], o[0][i])
            if (len(dtype) == 3) and (dtype[1] == "S4"):
                setattr(self, dtype[0], "".join([x.decode(unicode) for x in o[0][i]]))
        return


@dataclass
class FrequencyType:
    file_id: str = ""
    base_start: float = 0.0
    base_end: float = 0.0
    base_steps: int = 0
    tune_type: int = 0
    base_table: List[float] = field(default_factory=lambda: [0.0] * 8192)
    linear_step: float = 0.0
    log_step: float = 0.0
    freq_table_id: str = ""
    tune_steps: int = 0
    pulse_count: int = 0
    pulse_pattern: List[int] = field(default_factory=lambda: [0] * 256)
    pulse_offset: float = 0.0
    ramp_steps: int = 0
    ramp_repeats: int = 0
    drive_table: List[float] = field(default_factory=lambda: [0.0] * 8192)
    user: str = ""

    def read_frequency(self, fname: str, unicode: str = "latin-1") -> None:
        # Load all Frequency Type parameters
        o = np.memmap(
            fname,
            dtype=np.dtype(Frequency_default_factory),
            mode="r",
            offset=20324,
            shape=(1,),
        )
        for i, dtype in enumerate(Frequency_default_factory):
            setattr(self, dtype[0], o[0][i])
            if (len(dtype) == 3) and (dtype[1] == "S4"):
                setattr(
                    self,
                    dtype[0],
                    "".join([x.decode(unicode) for x in o[0][i]]),
                )
        return


@dataclass
class RecieverType:
    file_id: str = ""
    rx_chan: int = 0
    rx_map: List[int] = field(default_factory=lambda: [0] * 16)
    word_format: int = 0
    cic2_dec: int = 0
    cic2_interp: int = 0
    cic2_scale: int = 0
    cic5_dec: int = 0
    cic5_scale: int = 0
    rcf_type: str = ""
    rcf_dec: int = 0
    rcf_taps: int = 0
    coefficients: List[int] = field(default_factory=lambda: [0] * 160)
    analog_delay: float = 0.0
    user: str = ""

    def read_reciever(self, fname: str, unicode="latin-1") -> None:
        # Load all Reciever Type parameters
        o = np.memmap(
            fname,
            dtype=np.dtype(Reciever_default_factory),
            mode="r",
            offset=87184,
            shape=(1,),
        )
        for i, dtype in enumerate(Reciever_default_factory):
            setattr(self, dtype[0], o[0][i])
            if (len(dtype) == 3) and (dtype[1] == "S4"):
                setattr(self, dtype[0], "".join([x.decode(unicode) for x in o[0][i]]))
        return


@dataclass
class ExciterType:
    file_id: str = ""
    cic_scale: int = 0
    cic2_dec: int = 0
    cic2_interp: int = 0
    cic5_interp: int = 0
    rcf_type: str = ""
    rcf_taps: int = 0
    rcf_taps_phase: int = 0
    coefficients: List[int] = field(default_factory=lambda: [0] * 256)
    analog_delay: float = 0.0
    user: str = ""

    def read_exciter(self, fname: str, unicode="latin-1") -> None:
        # Load all Exciter Type parameters
        o = np.memmap(
            fname,
            dtype=np.dtype(Exciter_default_factory),
            mode="r",
            offset=88152,
            shape=(1,),
        )
        for i, dtype in enumerate(Exciter_default_factory):
            setattr(self, dtype[0], o[0][i])
            if (len(dtype) == 3) and (dtype[1] == "S4"):
                setattr(self, dtype[0], "".join([x.decode(unicode) for x in o[0][i]]))
        return


@dataclass
class MonitorType:
    balun_currents: List[int] = field(default_factory=lambda: [0] * 8)
    balun_status: List[int] = field(default_factory=lambda: [0] * 8)
    front_end_status: List[int] = field(default_factory=lambda: [0] * 8)
    receiver_status: List[int] = field(default_factory=lambda: [0] * 8)
    exciter_status: List[int] = field(default_factory=lambda: [0] * 2)
    user: str = ""

    def read_monitor(self, fname: str, unicode="latin-1") -> None:
        # Load all Frequency Type parameters
        o = np.memmap(
            fname,
            dtype=np.dtype(Monitor_default_factory),
            mode="r",
            offset=89428,
            shape=(1,),
        )
        for i, dtype in enumerate(Monitor_default_factory):
            setattr(self, dtype[0], o[0][i])
            if (len(dtype) == 3) and (dtype[1] == "S4"):
                setattr(self, dtype[0], "".join([x.decode(unicode) for x in o[0][i]]))
        return


@dataclass
class SctType:
    magic: int = 0x51495200
    sounding_table_size: int = 0
    pulse_table_size: int = 0
    raw_data_size: int = 0
    struct_version: float = 1.20
    start_year: int = 1970
    start_daynumber: int = 1
    start_month: int = 1
    start_day: int = 1
    start_hour: int = 0
    start_minute: int = 0
    start_second: int = 0
    start_epoch: int = 0
    readme: str = ""
    decimation_method: int = 0
    decimation_threshold: float = 0.0
    user: str = ""
    station: StationType = field(default_factory=StationType)
    timing: TimingType = field(default_factory=TimingType)
    frequency: FrequencyType = field(default_factory=FrequencyType)
    receiver: RecieverType = field(default_factory=RecieverType)
    exciter: ExciterType = field(default_factory=ExciterType)
    monitor: MonitorType = field(default_factory=MonitorType)

    def read_sct(self, fname: str, unicode="latin-1") -> None:
        logger.info(f"Reading SCT: {fname}")
        # Load all SCT Type parameters
        o = np.memmap(
            fname, dtype=np.dtype(SCT_default_factory), mode="r", offset=0, shape=(1,)
        )
        for i, dtype in enumerate(SCT_default_factory):
            setattr(self, dtype[0], o[0][i])
            if (len(dtype) == 3) and (dtype[1] == "S4"):
                setattr(self, dtype[0], "".join([x.decode(unicode) for x in o[0][i]]))
        self.station.read_station(fname, unicode)
        self.timing.read_timing(fname, unicode)
        self.frequency.read_frequency(fname, unicode)
        self.receiver.read_reciever(fname, unicode)
        self.exciter.read_exciter(fname, unicode)
        self.monitor.read_monitor(fname, unicode)

        self.fix_SCT_strings()
        return

    def fix_SCT_strings(self) -> None:
        logger.info("Fixing SCT strings...")
        self.user = trim_null(self.user)
        self.readme = trim_null(self.readme)

        self.station.file_id = trim_null(self.station.file_id)
        self.station.ursi_id = trim_null(self.station.ursi_id)
        self.station.rx_name = trim_null(self.station.rx_name)
        self.station.rx_antenna_type = trim_null(self.station.rx_antenna_type)
        self.station.tx_name = trim_null(self.station.tx_name)
        self.station.tx_antenna_type = trim_null(self.station.tx_antenna_type)
        self.station.ref_type = trim_null(self.station.ref_type)
        self.station.clock_type = trim_null(self.station.clock_type)

        self.timing.file_id = trim_null(self.timing.file_id)
        self.timing.data_wave_file = trim_null(self.timing.data_wave_file)
        self.timing.cal_wave_file = trim_null(self.timing.cal_wave_file)
        self.timing.user = trim_null(self.timing.user)

        self.frequency.file_id = trim_null(self.frequency.file_id)
        self.frequency.freq_table_id = trim_null(self.frequency.freq_table_id)
        self.frequency.user = trim_null(self.frequency.user)

        self.receiver.file_id = trim_null(self.receiver.file_id)
        self.receiver.rcf_type = trim_null(self.receiver.rcf_type)
        self.receiver.user = trim_null(self.receiver.user)
        self.exciter.file_id = trim_null(self.exciter.file_id)

        self.exciter.rcf_type = trim_null(self.exciter.rcf_type)
        self.exciter.user = trim_null(self.exciter.user)
        return

    def dump_sct(self, to_file: str = None) -> None:
        self.fix_SCT_strings()
        txt = "General:\n"
        txt += f"sct.magic: 0x{self.magic:X}\n"
        txt += f"sct.sounding_table_size: {self.sounding_table_size}\n"
        txt += f"sct.pulse_table_size: {self.pulse_table_size}\n"
        txt += f"sct.raw_data_size: {self.raw_data_size}\n"
        txt += f"sct.struct_version: {self.struct_version}\n"
        txt += f"sct.start_year: {self.start_year}\n"
        txt += f"sct.start_daynumber: {self.start_daynumber}\n"
        txt += f"sct.start_month: {self.start_month}\n"
        txt += f"sct.start_day: {self.start_day}\n"
        txt += f"sct.start_hour: {self.start_hour}\n"
        txt += f"sct.start_minute: {self.start_minute}\n"
        txt += f"sct.start_second: {self.start_second}\n"
        txt += f"sct.start_epoch: {self.start_epoch:.2f}\n"
        txt += f"sct.readme: {self.readme}\n"
        txt += f"sct.user: {self.user}\n"

        txt += "\nStation:\n"
        txt += f"sct.station.file_id: {self.station.file_id}\n"
        txt += f"sct.station.ursi_id: {self.station.ursi_id}\n"
        txt += f"sct.station.rx_name: {self.station.rx_name}\n"
        txt += f"sct.station.rx_latitude: {self.station.rx_latitude:.2f}\n"
        txt += f"sct.station.rx_longitude: {self.station.rx_longitude:.2f}\n"
        txt += f"sct.station.rx_altitude: {self.station.rx_altitude:.2f}\n"
        txt += f"sct.station.rx_count: {self.station.rx_count}\n"
        txt += f"sct.station.rx_antenna_type: {self.station.rx_antenna_type}\n"
        txt += (
            " rx_position [X Y Z]"
            + " rx_direction [X Y Z]"
            + " rx_height"
            + " rx_cable_length\n"
        )
        for pos, dir, ht, ln in zip(
            self.station.rx_position,
            self.station.rx_direction,
            self.station.rx_height,
            self.station.rx_cable_length,
        ):
            txt += f" {str(pos)} {str(pos)} {str(ht)} {str(ln)}\n"
        txt += f"sct.station.frontend_atten: {self.station.frontend_atten}\n"
        txt += f"sct.station.tx_name: {self.station.tx_name.strip()}\n"
        txt += f"sct.station.tx_latitude: {self.station.tx_latitude}\n"
        txt += f"sct.station.tx_longitude: {self.station.tx_longitude}\n"
        txt += f"sct.station.tx_altitude: {self.station.tx_altitude}\n"
        txt += f"sct.station.tx_vector: {self.station.tx_vector}\n"
        txt += f"sct.station.tx_height: {self.station.tx_height}\n"
        txt += f"sct.station.tx_cable_length: {self.station.tx_cable_length}\n"
        txt += f"sct.station.drive_band_count: {self.station.drive_band_count}\n"
        txt += f"sct.station.rf_control: {self.station.rf_control}\n"
        txt += f"sct.station.ref_type: {self.station.ref_type.strip()}\n"
        txt += f"sct.station.clock_type: {self.station.clock_type.strip()}\n"
        txt += f"sct.station.user: {self.station.user.strip()}\n"

        txt += "\nTiming:\n"
        txt += f"sct.timing.file_id: {self.timing.file_id.strip()}\n"
        txt += f"sct.timing.pri: {self.timing.pri}\n"
        txt += f"sct.timing.pri_count: {self.timing.pri_count}\n"
        txt += f"sct.timing.ionogram_count: {self.timing.ionogram_count}\n"
        txt += f"sct.timing.holdoff: {self.timing.holdoff}\n"
        txt += f"sct.timing.range_gate_offset: {self.timing.range_gate_offset}\n"
        txt += f"sct.timing.gate_count: {self.timing.gate_count}\n"
        txt += f"sct.timing.gate_start: {self.timing.gate_start}\n"
        txt += f"sct.timing.gate_end: {self.timing.gate_end}\n"
        txt += f"sct.timing.gate_step: {self.timing.gate_step}\n"
        txt += f"sct.timing.data_start: {self.timing.data_start}\n"
        txt += f"sct.timing.data_width: {self.timing.data_width}\n"
        txt += f"sct.timing.data_baud_count: {self.timing.data_baud_count}\n"
        txt += f"sct.timing.data_wave_file: {self.timing.data_wave_file.strip()}\n"
        txt += f"sct.timing.data_baud: {','.join([str(c) for c in self.timing.data_baud])}\n"
        txt += f"sct.timing.data_pairs: {self.timing.data_pairs}\n"
        txt += f"sct.timing.cal_start: {self.timing.cal_start}\n"
        txt += f"sct.timing.cal_width: {self.timing.cal_width}\n"
        txt += f"sct.timing.cal_baud_count: {self.timing.cal_baud_count}\n"
        txt += f"sct.timing.cal_wave_file: {self.timing.cal_wave_file.strip()}\n"
        txt += (
            f"sct.timing.cal_baud: {','.join([str(c) for c in self.timing.cal_baud])}\n"
        )
        txt += f"sct.timing.cal_pairs: {self.timing.cal_pairs}\n"
        txt += f"sct.timing.user: {self.timing.user.strip()}\n"

        txt += "\nFrequency:\n"
        txt += f"sct.frequency.file_id: {self.frequency.file_id.strip()}\n"
        txt += f"sct.frequency.base_start: {self.frequency.base_start}\n"
        txt += f"sct.frequency.base_end: {self.frequency.base_end}\n"
        txt += f"sct.frequency.base_steps: {self.frequency.base_steps}\n"
        txt += f"sct.frequency.tune_type: {self.frequency.tune_type}\n"
        txt += f"sct.frequency.base_table: {','.join([str(c) for c in self.frequency.base_table])}\n"
        txt += f"sct.frequency.linear_step: {self.frequency.linear_step}\n"
        txt += f"sct.frequency.log_step: {self.frequency.log_step}\n"
        txt += f"sct.frequency.freq_table_id: {self.frequency.freq_table_id.strip()}\n"
        txt += f"sct.frequency.tune_steps: {self.frequency.tune_steps}\n"
        txt += f"sct.frequency.pulse_count: {self.frequency.pulse_count}\n"
        txt += f"sct.frequency.pulse_pattern: {','.join([str(c) for c in self.frequency.pulse_pattern])}\n"
        txt += f"sct.frequency.pulse_offset: {self.frequency.pulse_offset}\n"
        txt += f"sct.frequency.ramp_steps: {self.frequency.ramp_steps}\n"
        txt += f"sct.frequency.drive_table: {','.join([str(c) for c in self.frequency.drive_table])}\n"
        txt += f"sct.frequency.user: {self.frequency.user.strip()}\n"

        txt += "\nReciever:\n"
        txt += f"sct.receiver.file_id: {self.receiver.file_id.strip()}\n"
        txt += f"sct.receiver.rx_chan: {self.receiver.rx_chan}\n"
        txt += f"sct.receiver.word_format: {self.receiver.word_format}\n"
        txt += f"sct.receiver.cic2_dec: {self.receiver.cic2_dec}\n"
        txt += f"sct.receiver.cic2_interp: {self.receiver.cic2_interp}\n"
        txt += f"sct.receiver.cic2_scale: {self.receiver.cic2_scale}\n"
        txt += f"sct.receiver.cic5_dec: {self.receiver.cic5_dec}\n"
        txt += f"sct.receiver.cic5_scale: {self.receiver.cic5_scale}\n"
        txt += f"sct.receiver.rcf_type: {self.receiver.rcf_type}\n"
        txt += f"sct.receiver.rcf_dec: {self.receiver.rcf_dec}\n"
        txt += f"sct.receiver.rcf_taps: {self.receiver.rcf_taps}\n"
        txt += f"sct.receiver.analog_delay: {self.receiver.analog_delay}\n"
        txt += f"sct.receiver.coefficients: {','.join([str(c) for c in self.receiver.coefficients])}\n"
        txt += f"sct.receiver.coefficients: {','.join([str(c) for c in self.receiver.rx_map])}\n"
        txt += f"sct.receiver.user: {self.receiver.user.strip()}\n"

        txt += "\nExciter:\n"
        txt += f"sct.exciter.file_id: {self.exciter.file_id.strip()}\n"
        txt += f"sct.exciter.cic_scale: {self.exciter.cic_scale}\n"
        txt += f"sct.exciter.cic2_dec: {self.exciter.cic2_dec}\n"
        txt += f"sct.exciter.cic2_interp: {self.exciter.cic2_interp}\n"
        txt += f"sct.exciter.cic5_interp: {self.exciter.cic5_interp}\n"
        txt += f"sct.exciter.rcf_type: {self.exciter.rcf_type}\n"
        txt += f"sct.exciter.rcf_taps: {self.exciter.rcf_taps}\n"
        txt += f"sct.exciter.rcf_taps_phase: {self.exciter.rcf_taps_phase}\n"
        txt += f"sct.exciter.analog_delay: {self.exciter.analog_delay}\n"
        txt += f"sct.exciter.coefficients: {','.join([str(c) for c in self.exciter.coefficients])}\n"
        txt += f"sct.exciter.user: {self.exciter.user.strip()}\n"

        txt += "\nMonitor:\n"
        txt += f"sct.monitor.balun_status: {self.monitor.balun_status}\n"
        txt += f"sct.monitor.balun_currents: {self.monitor.balun_currents}\n"
        txt += f"sct.monitor.front_end_status: {self.monitor.front_end_status}\n"
        txt += f"sct.monitor.receiver_status: {self.monitor.receiver_status}\n"
        txt += f"sct.monitor.exciter_status: {self.monitor.exciter_status}\n"
        txt += f"sct.monitor.user: {self.monitor.user.strip()}"

        if to_file:
            with open(to_file, "w") as f:
                f.write(txt)
        else:
            logger.info(f"# SCT: \n {txt}")
        return
