from cmath import rect  # For complex numbers
from dataclasses import dataclass, field
from typing import List

from loguru import logger

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
    rx_antenna_type: List[str] = field(default_factory=lambda: [""] * 32)
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


@dataclass
class MonitorType:
    balun_currents: List[int] = field(default_factory=lambda: [0] * 8)
    balun_status: List[int] = field(default_factory=lambda: [0] * 8)
    front_end_status: List[int] = field(default_factory=lambda: [0] * 8)
    receiver_status: List[int] = field(default_factory=lambda: [0] * 8)
    exciter_status: List[int] = field(default_factory=lambda: [0] * 2)
    user: str = ""


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

    def fix_SCT_strings(self) -> None:
        logger.info("Fixing SCT strings...")
        self.user = trim_null(self.user)
        self.readme = trim_null(self.readme)

        self.station.file_id = trim_null(self.station.file_id)
        self.station.ursi_id = trim_null(self.station.ursi_id)
        self.station.rx_name = trim_null(self.station.rx_name)
        self.station.rx_antenna_type = [
            trim_null(rx_antenna_type)
            for rx_antenna_type in self.station.rx_antenna_type
        ]
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

    def read_pct(self) -> None:
        logger.info("Reading SCT strings...")
        return

    def dump_sct(self) -> None:
        self.fix_SCT_strings()
        txt = "General:"
        txt += f"sct.magic: 0x{self.magic:X}"
        txt += f"sct.sounding_table_size: {self.sounding_table_size}"
        txt += f"sct.pulse_table_size: {self.pulse_table_size}"
        txt += f"sct.raw_data_size: {self.raw_data_size}"
        txt += f"sct.struct_version: {self.struct_version}"
        txt += f"sct.start_year: {self.start_year}"
        txt += f"sct.start_daynumber: {self.start_daynumber}"
        txt += f"sct.start_month: {self.start_month}"
        txt += f"sct.start_day: {self.start_day}"
        txt += f"sct.start_hour: {self.start_hour}"
        txt += f"sct.start_minute: {self.start_minute}"
        txt += f"sct.start_second: {self.start_second}"
        txt += f"sct.start_epoch: {self.start_epoch:.2f}"
        txt += f"sct.readme: {self.readme}"
        txt += f"sct.user: {self.user}\n"

        txt += "Station:"
        txt += f"sct.station.file_id: {self.station.file_id}"
        txt += f"sct.station.ursi_id: {self.station.ursi_id}"
        txt += f"sct.station.rx_name: {self.station.rx_name}"
        txt += f"sct.station.rx_latitude: {self.station.rx_latitude:.2f}"
        txt += f"sct.station.rx_longitude: {self.station.rx_longitude:.2f}"
        txt += f"sct.station.rx_altitude: {self.station.rx_altitude:.2f}"
        txt += f"sct.station.rx_count: {self.station.rx_count}"

        # Todo

        logger.info(f"# SCT: \n {txt}")
        return
