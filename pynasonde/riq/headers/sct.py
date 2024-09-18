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

        txt += "Station:\n"
        txt += f"sct.station.file_id: {self.station.file_id}\n"
        txt += f"sct.station.ursi_id: {self.station.ursi_id}\n"
        txt += f"sct.station.rx_name: {self.station.rx_name}\n"
        txt += f"sct.station.rx_latitude: {self.station.rx_latitude:.2f}\n"
        txt += f"sct.station.rx_longitude: {self.station.rx_longitude:.2f}\n"
        txt += f"sct.station.rx_altitude: {self.station.rx_altitude:.2f}\n"
        txt += f"sct.station.rx_count: {self.station.rx_count}\n"
        txt += (
            "rx_antenna_type",
            "rx_position X Y Z",
            "rx_direction X Y Z",
            "rx_height",
            "rx_cable_length\n",
        )
        k = max(1, min(self.station.rx_count, 32))
        if self.verbose:
            k = 32
        for j in range(1, k + 1):
            txt += (
                self.station.rx_antenna_type[j - 1]
                + self.station.rx_position[:, j - 1]
                + self.station.rx_direction[:, j - 1]
                + self.station.rx_height[j - 1]
                + self.station.rx_cable_length[j - 1]
            )
        txt += f"sct.station.frontend_atten: {self.station.frontend_atten}\n"
        txt += f"sct.station.tx_name: {self.station.tx_name.strip()}\n"
        txt += f"sct.station.tx_latitude: {self.station.tx_latitude}\n"
        txt += f"sct.station.tx_longitude: {self.station.tx_longitude}\n"
        txt += f"sct.station.tx_altitude: {self.station.tx_altitude}\n"
        txt += f"sct.station.tx_vector: {self.station.tx_vector}\n"
        txt += f"sct.station.tx_height: {self.station.tx_height}\n"
        txt += f"sct.station.tx_cable_length: {self.station.tx_cable_length}\n"
        txt += f"sct.station.drive_band_count: {self.station.drive_band_count}\n"
        txt += "drive_band_bounds", "drive_band_bounds", "drive_band_atten\n"
        k = max(1, min(self.station.drive_band_count, 64))
        if self.verbose:
            k = 64
        for j in range(1, k + 1):
            txt += (
                self.station.drive_band_bounds[:, j - 1]
                + self.station.drive_band_atten[j - 1]
            )
        txt += f"sct.station.rf_control: {self.station.rf_control}\n"
        txt += f"sct.station.ref_type: {self.station.ref_type.strip()}\n"
        txt += f"sct.station.clock_type: {self.station.clock_type.strip()}\n"
        txt += f"sct.station.user: {self.station.user.strip()}\n"

        txt += "Timing:\n"
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
        txt += "sct.timing.data_baud\n"
        k = max(1, min(self.timing.data_baud_count, 1024))
        if self.verbose:
            k = 1024
        for i in range(1, k + 1):
            txt += f"{i}, {self.timing.data_baud[i - 1]}\n"
        txt += f"sct.timing.data_pairs: {self.timing.data_pairs}\n"
        txt += f"sct.timing.cal_start: {self.timing.cal_start}\n"
        txt += f"sct.timing.cal_width: {self.timing.cal_width}\n"
        txt += f"sct.timing.cal_baud_count: {self.timing.cal_baud_count}\n"
        txt += f"sct.timing.cal_wave_file: {self.timing.cal_wave_file.strip()}\n"
        txt += "sct.timing.cal_baud\n"
        k = max(1, min(self.timing.cal_baud_count, 1024))
        if self.verbose:
            k = 1024
        for i in range(1, k + 1):
            txt += f"{i}, {self.timing.cal_baud[i - 1]}\n"
        txt += f"sct.timing.cal_pairs: {self.timing.cal_pairs}\n"
        txt += f"sct.timing.user: {self.timing.user.strip()}\n"

        txt += "Frequency:\n"
        txt += f"sct.frequency.file_id: {self.frequency.file_id.strip()}\n"
        txt += f"sct.frequency.base_start: {self.frequency.base_start}\n"
        txt += f"sct.frequency.base_end: {self.frequency.base_end}\n"
        txt += f"sct.frequency.base_steps: {self.frequency.base_steps}\n"
        txt += f"sct.frequency.tune_type: {self.frequency.tune_type}\n"
        txt += "sct.frequency.base_table\n"
        k = max(1, min(self.frequency.base_steps, 8192))
        if self.verbose:
            k = 8192
        for i in range(1, k + 1):
            txt += f"{i}, {self.frequency.base_table[i - 1]}\n"
        txt += f"sct.frequency.linear_step: {self.frequency.linear_step}\n"
        txt += f"sct.frequency.log_step: {self.frequency.log_step}\n"
        txt += f"sct.frequency.freq_table_id: {self.frequency.freq_table_id.strip()}\n"
        txt += f"sct.frequency.tune_steps: {self.frequency.tune_steps}\n"
        txt += f"sct.frequency.pulse_count: {self.frequency.pulse_count}\n"
        txt += "sct.frequency.pulse_pattern\n"
        k = max(1, min(self.frequency.pulse_count, 256))
        if self.verbose:
            k = 256
        for i in range(1, k + 1):
            txt += f"{i}, {self.frequency.pulse_pattern[i - 1]}\n"
        txt += f"sct.frequency.pulse_offset: {self.frequency.pulse_offset}\n"
        txt += f"sct.frequency.ramp_steps: {self.frequency.ramp_steps}\n"
        txt += (
            f"sct.frequency.freq_hop_table: {self.frequency.freq_hop_table.strip()}\n"
        )
        txt += f"sct.frequency.user: {self.frequency.user.strip()}\n"

        txt += "Reciever:\n"
        txt += f"sct.receiver.file_id: {self.receiver.file_id.strip()}\n"
        txt += f"sct.receiver.sample_rate: {self.receiver.sample_rate}\n"
        txt += f"sct.receiver.sample_width: {self.receiver.sample_width}\n"
        txt += f"sct.receiver.decimation: {self.receiver.decimation}\n"
        txt += f"sct.receiver.frontend_id: {self.receiver.frontend_id.strip()}\n"
        txt += f"sct.receiver.rx_attenuator: {self.receiver.rx_attenuator}\n"
        txt += f"sct.receiver.calibration: {self.receiver.calibration}\n"
        txt += f"sct.receiver.coherent_channel: {self.receiver.coherent_channel}\n"
        txt += f"sct.receiver.user: {self.receiver.user.strip()}\n"

        txt += "Exciter:\n"
        txt += f"sct.exciter.file_id: {self.exciter.file_id.strip()}\n"
        txt += f"sct.exciter.exciter_id: {self.exciter.exciter_id.strip()}\n"
        txt += f"sct.exciter.nominal_freq: {self.exciter.nominal_freq}\n"
        txt += f"sct.exciter.rf_attenuation: {self.exciter.rf_attenuation}\n"
        txt += f"sct.exciter.user: {self.exciter.user.strip()}\n"

        txt += "Monitor:\n"
        txt += f"sct.monitor.file_id: {self.monitor.file_id.strip()}\n"
        txt += f"sct.monitor.start_mode: {self.monitor.start_mode}\n"
        txt += f"sct.monitor.end_mode: {self.monitor.end_mode}\n"
        txt += f"sct.monitor.tx_power: {self.monitor.tx_power}"
        txt += f"sct.monitor.calibration: {self.monitor.calibration}"
        txt += f"sct.monitor.drive_value: {self.monitor.drive_value}"
        txt += f"sct.monitor.user: {self.monitor.user.strip()}"

        logger.info(f"# SCT: \n {txt}")
        return
