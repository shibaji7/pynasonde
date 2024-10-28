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
    file_id: str = ""  # Name of station settings file
    ursi_id: str = ""  # URSI standard station ID code
    rx_name: str = ""  # Receiver Station Name
    rx_latitude: np.float64 = (
        0.0  # Position of the Receive array reference point [degrees North]
    )
    rx_longitude: np.float64 = (
        0.0  # Position of the Receive array reference point [degrees East]
    )
    rx_altitude: np.float64 = 0.0  # Meters above mean sea level
    rx_count: np.int32 = 0  # Number of defined receive antennas
    rx_antenna_type: List[str] = field(
        default_factory=lambda: [""] * 16
    )  # Rx antenna type text descriptors
    rx_position: List[List[float]] = field(
        default_factory=lambda: [[0.0] * 32 for _ in range(3)]
    )  # X,Y,Z = (East,North,Up) Positon [m] of each Rx
    rx_direction: List[List[float]] = field(
        default_factory=lambda: [[0.0] * 32 for _ in range(3)]
    )  # X,Y,Z = (East,North,Up) Direction of each Rx
    rx_height: List[float] = field(
        default_factory=lambda: [0.0] * 32
    )  # Height above ground [m]
    rx_cable_length: List[float] = field(
        default_factory=lambda: [0.0] * 32
    )  # Physical length of receive cables [m]
    frontend_atten: np.float64 = 0.0  # Front End attenuator setting
    tx_name: str = ""  # Transmitter Station Name
    tx_latitude: np.float64 = (
        0.0  # Position of the Transmit Antenna reference point [degrees North]
    )
    tx_longitude: np.float64 = (
        0.0  # Position of the Transmit Antenna reference point [degrees East]
    )
    tx_altitude: np.float64 = 0.0  # Meters above mean sea level
    tx_antenna_type: str = ""  # Tx antenna type text descriptors
    tx_vector: List[float] = field(
        default_factory=lambda: [0.0] * 3
    )  # Tx antenna direction vector [m]
    tx_height: np.float64 = 0.0  # Antenna height above reference ground [m]
    tx_cable_length: np.float64 = 0.0  # Physical length of transmit cables [m]
    drive_band_count: np.int32 = 0  # Number of antenna drive bands
    drive_band_bounds: List[List[float]] = field(
        default_factory=lambda: [[0.0] * 64 for _ in range(2)]
    )  # Drive bands start/stop in kHz
    drive_band_atten: List[float] = field(
        default_factory=lambda: [0.0] * 64
    )  # Antenna drive atteunuation in dB
    rf_control: np.int32 = (
        -1
    )  # -1 = none, 0 = drive/quiet, 1 = full, 2 = only quiet, 3 = only atten
    ref_type: str = ""  # Type of reference oscillator
    clock_type: str = ""  # Source of absoulte UT timing
    user: str = ""  # Spare space for user-defined information

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
                setattr(
                    self,
                    dtype[0],
                    (
                        "".join([char.decode(unicode) for char in o[0][i]])
                        if len(dtype[2]) == 1
                        else [
                            "".join([char.decode(unicode) for char in chars])
                            for chars in o[0][i]
                        ]
                    ),
                )
        return


@dataclass
class TimingType:
    """
    Time values are in microseonds unless otherwise indicated
    """

    file_id: str = ""  # Name of the timing settings file
    pri: np.float64 = 0.0  # Pulse Repetition Interval (PRI) (microseconds)
    pri_count: np.int32 = 0  # number of PRI's in the measurement
    ionogram_count: np.int32 = 0  # Repeat count for ionogram within same data file
    holdoff: np.float64 = 0.0  # Time between GPS 1 pps and start
    range_gate_offset: np.float64 = 0.0  # True range to gate 0
    gate_count: np.int32 = 0  # Number of range gates, adjusted up for USB blocks
    gate_start: np.float64 = 0.0  # Start gate placement [us], adjusted
    gate_end: np.float64 = 0.0  # End gate placement [us], adjusted
    gate_step: np.float64 = 0.0  # Range delta [us]
    data_start: np.float64 = 0.0  # Data range placement start [us]
    data_width: np.float64 = 0.0  # Data pulse baud width [us]
    data_baud_count: np.int32 = 0  # Data pulse baud count
    data_wave_file: str = ""  # Data baud pattern file name
    data_baud: List[complex] = field(
        default_factory=lambda: [complex(0.0, 0.0)] * 1024
    )  # Data waveform baud pattern
    data_pairs: np.int32 = 0  # Number of IQ pairs in waveform memory
    cal_start: np.float64 = 0.0  # Cal range placement start [us]
    cal_width: np.float64 = 0.0  # Cal pulse baud width [us]
    cal_baud_count: np.int32 = 0  # Cal pulse baud count
    cal_wave_file: str = ""  # Alternative baud pattern file name
    cal_baud: List[complex] = field(
        default_factory=lambda: [complex(0.0, 0.0)] * 1024
    )  # Cal waveform baud pattern
    cal_pairs: np.int32 = 0  # Number of IQ pairs in waveform memory
    user: str = ""  # Spare space for user-defined information

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
    """
    Values are in kilohertz unless otherwise indicated
    """

    file_id: str = ""  # Frequency settings file
    base_start: np.float64 = 0.0  # Initial base frequency
    base_end: np.float64 = 0.0  # Final base frequency
    base_steps: np.int32 = 0  # Number of base frequencies
    tune_type: np.int32 = (
        0  # Tuning type flag:  1=log, 2=linear, 3=table, 4=Log+Fixed ShuffleMode
    )
    base_table: List[float] = field(
        default_factory=lambda: [0.0] * 8192
    )  # Nominal or Base frequency table
    linear_step: np.float64 = 0.0  # Linear frequency step [kHz]
    log_step: np.float64 = 0.0  # Log frequency step, [percent]
    freq_table_id: str = ""  # Manual tuning table filename
    tune_steps: np.int32 = 0  # All frequencies pre-ramp repeats
    pulse_count: np.int32 = 0  # Pulset frequency vector length
    pulse_pattern: List[int] = field(
        default_factory=lambda: [0] * 256
    )  # Pulse_pattern ! Pulset frequency vector
    pulse_offset: np.float64 = 0.0  # Pulset offset [kHz]
    ramp_steps: np.int32 = (
        0  # Pulsets per B-mode ramp (ramp length, base freqs per B-block)
    )
    ramp_repeats: np.int32 = 0  # Repeat count of B-mode ramps
    drive_table: List[float] = field(
        default_factory=lambda: [0.0] * 8192
    )  # Base frequencies attenuation/silent table
    user: str = ""  # Spare space for user-defined information

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
    file_id: str = ""  # Frequency settings file
    rx_chan: np.int32 = 0  # Number of receivers
    rx_map: List[int] = field(
        default_factory=lambda: [0] * 16
    )  # Receiver-to-antenna mapping
    word_format: np.int32 = (
        0  # 0 = big endian fixed, 1 = little endian, 2 = floating_point, 3=32 bit little endian integer (v2.z)
    )
    cic2_dec: np.int32 = 0  # DDC filter block
    cic2_interp: np.int32 = 0  # DDC filter block
    cic2_scale: np.int32 = 0  # DDC filter block
    cic5_dec: np.int32 = 0  # DDC filter block
    cic5_scale: np.int32 = 0  # DDC filter block
    rcf_type: str = ""  # Text descriptor of FIR filter block
    rcf_dec: np.int32 = 0  # Decimation factor for FIR filter block
    rcf_taps: np.int32 = 0  # Number of taps in FIR filter block
    coefficients: List[int] = field(
        default_factory=lambda: [0] * 160
    )  # Receiver filter coefficients
    analog_delay: np.float64 = 0.0  # Analog delay of receiver, us
    user: str = ""  # Spare space for user-defined information

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
    file_id: str = ""  # Frequency settings file
    cic_scale: np.int32 = 0  # DUC filter block
    cic2_dec: np.int32 = 0  # DUC filter block
    cic2_interp: np.int32 = 0  # DUC filter block
    cic5_interp: np.int32 = 0  # DUC filter block
    rcf_type: str = ""  # Text descriptor of FIR filter block
    rcf_taps: np.int32 = 0  # Number of taps in FIR filter block
    rcf_taps_phase: np.int32 = 0  # Number of taps in FIR filter block
    coefficients: List[int] = field(
        default_factory=lambda: [0] * 256
    )  # Receiver filter coefficients
    analog_delay: np.float64 = 0.0  # Analog delay of exciter/transmitter, us
    user: str = ""  # Spare space for user-defined information

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
    balun_currents: List[int] = field(
        default_factory=lambda: [0] * 8
    )  # As read prior to ionogram
    balun_status: List[int] = field(
        default_factory=lambda: [0] * 8
    )  # As read prior to ionogram
    front_end_status: List[int] = field(
        default_factory=lambda: [0] * 8
    )  # As read prior to ionogram
    receiver_status: List[int] = field(
        default_factory=lambda: [0] * 8
    )  # As read prior to ionogram
    exciter_status: List[int] = field(
        default_factory=lambda: [0] * 2
    )  # As read prior to ionogram
    user: str = ""  # Spare space for user-defined information

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
    magic: np.int32 = (
        0x51495200  # Magic number 0x51495200 (/nullRIQ) {POSSIBLY BYTE REVERSED}
    )
    sounding_table_size: np.int32 = (
        0  # Bytes in sounder configuration structure (this file)
    )
    pulse_table_size: np.int32 = 0  # Bytes in pulse configuration structure
    raw_data_size: np.int32 = 0  # Bytes in raw data block (one PRI)
    struct_version: np.float64 = 1.20  # Format Version Number. Currently 1.2
    start_year: np.int32 = 1970  # Start Time Elements of the ionogram (Universal Time)
    start_daynumber: np.int32 = 1
    start_month: np.int32 = 1
    start_day: np.int32 = 1
    start_hour: np.int32 = 0
    start_minute: np.int32 = 0
    start_second: np.int32 = 0
    start_epoch: np.int32 = 0  # Epoch time of the measurement start.
    readme: str = ""  # Operator comment on this measurement
    decimation_method: np.int32 = 0  # If processed, 0=no process (raw data)
    decimation_threshold: np.float64 = (
        0.0  # If processed, the treshold value for the given method
    )
    user: str = ""  # user-defined
    station: StationType = field(
        default_factory=StationType
    )  # Station info substructure
    timing: TimingType = field(default_factory=TimingType)  # Radar timing substruture
    frequency: FrequencyType = field(
        default_factory=FrequencyType
    )  # Frequency sweep substructure
    receiver: RecieverType = field(
        default_factory=RecieverType
    )  # Receiver settings substructure
    exciter: ExciterType = field(
        default_factory=ExciterType
    )  # Exciter settings substructure
    monitor: MonitorType = field(
        default_factory=MonitorType
    )  # Built In Test values substructure

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

        # self.fix_SCT_strings()
        return

    def fix_SCT_strings(self) -> None:
        logger.info("Fixing SCT strings...")
        self.user = trim_null(self.user)
        self.readme = trim_null(self.readme)

        self.station.file_id = trim_null(self.station.file_id)
        self.station.ursi_id = trim_null(self.station.ursi_id)
        self.station.rx_name = trim_null(self.station.rx_name)
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
        # self.fix_SCT_strings()
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
        txt += (
            "sct.station:\n"
            + " rx_antenna_type rx_position[X Y Z]"
            + " rx_direction[X Y Z]"
            + " rx_height"
            + " rx_cable_length\n"
        )
        for atype, pos, dir, ht, ln in zip(
            self.station.rx_antenna_type[: self.station.rx_count],
            self.station.rx_position[: self.station.rx_count],
            self.station.rx_direction[: self.station.rx_count],
            self.station.rx_height[: self.station.rx_count],
            self.station.rx_cable_length[: self.station.rx_count],
        ):
            txt += f" {atype}\t{str(pos)}\t{str(dir)}\t{str(ht)}\t{str(ln)}\n"
        txt += f"sct.station.frontend_atten: {self.station.frontend_atten}\n"
        txt += f"sct.station.tx_name: {self.station.tx_name.strip()}\n"
        txt += f"sct.station.tx_latitude: {self.station.tx_latitude}\n"
        txt += f"sct.station.tx_longitude: {self.station.tx_longitude}\n"
        txt += f"sct.station.tx_altitude: {self.station.tx_altitude}\n"
        txt += f"sct.station.tx_vector: {self.station.tx_vector}\n"
        txt += f"sct.station.tx_height: {self.station.tx_height}\n"
        txt += f"sct.station.tx_cable_length: {self.station.tx_cable_length}\n"
        txt += f"sct.station.drive_band_count: {self.station.drive_band_count}\n"
        txt += (
            "sct.station:\n" + " drive_band_bounds drive_band_bounds drive_band_atten\n"
        )
        for bounds, attn in zip(
            self.station.drive_band_bounds[: self.station.drive_band_count],
            self.station.drive_band_atten[: self.station.drive_band_count],
        ):
            txt += f" {bounds[0]}\t{bounds[1]}\t{attn}\n"
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
        t = "\n".join(
            [
                f"{ix+1}: {tm}"
                for ix, tm in enumerate(
                    self.timing.data_baud[
                        : max(1, min(self.timing.data_baud_count, 1024))
                    ]
                )
            ]
        )
        txt += f"sct.timing.data_baud:\n {t}\n"
        txt += f"sct.timing.data_pairs: {self.timing.data_pairs}\n"
        txt += f"sct.timing.cal_start: {self.timing.cal_start}\n"
        txt += f"sct.timing.cal_width: {self.timing.cal_width}\n"
        txt += f"sct.timing.cal_baud_count: {self.timing.cal_baud_count}\n"
        txt += f"sct.timing.cal_wave_file: {self.timing.cal_wave_file.strip()}\n"
        t = "\n".join(
            [
                f"{ix+1}: {tm}"
                for ix, tm in enumerate(
                    self.timing.cal_baud[
                        : max(1, min(self.timing.cal_baud_count, 1024))
                    ]
                )
            ]
        )
        txt += f"sct.timing.cal_baud:\n {t}\n"
        txt += f"sct.timing.cal_pairs: {self.timing.cal_pairs}\n"
        txt += f"sct.timing.user: {self.timing.user.strip()}\n"

        txt += "\nFrequency:\n"
        txt += f"sct.frequency.file_id: {self.frequency.file_id.strip()}\n"
        txt += f"sct.frequency.base_start: {self.frequency.base_start}\n"
        txt += f"sct.frequency.base_end: {self.frequency.base_end}\n"
        txt += f"sct.frequency.base_steps: {self.frequency.base_steps}\n"
        txt += f"sct.frequency.tune_type: {self.frequency.tune_type}\n"
        t = "\n ".join(
            [
                f"{ix+1}: {fq}"
                for ix, fq in enumerate(
                    self.frequency.base_table[
                        : max(1, min(self.frequency.base_steps, 8182))
                    ]
                )
            ]
        )
        txt += f"sct.frequency.base_table:\n {t}\n"
        txt += f"sct.frequency.linear_step: {self.frequency.linear_step}\n"
        txt += f"sct.frequency.log_step: {self.frequency.log_step}\n"
        txt += f"sct.frequency.freq_table_id: {self.frequency.freq_table_id.strip()}\n"
        txt += f"sct.frequency.tune_steps: {self.frequency.tune_steps}\n"
        txt += f"sct.frequency.pulse_count: {self.frequency.pulse_count}\n"
        t = "\n ".join(
            [
                f"{ix+1}: {fq}"
                for ix, fq in enumerate(
                    self.frequency.pulse_pattern[
                        : max(1, min(self.frequency.pulse_count, 256))
                    ]
                )
            ]
        )
        txt += f"sct.frequency.pulse_pattern: \n{t}\n"
        txt += f"sct.frequency.pulse_offset: {self.frequency.pulse_offset}\n"
        txt += f"sct.frequency.ramp_steps: {self.frequency.ramp_steps}\n"
        t = "\n ".join(
            [
                f"{ix+1}: {fq}"
                for ix, fq in enumerate(
                    self.frequency.drive_table[
                        : max(1, min(self.frequency.base_steps, 8192))
                    ]
                )
            ]
        )
        txt += f"sct.frequency.drive_table:\n {t}\n"
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
        t = "\n ".join(
            [
                f"{ix+1}: {fq}"
                for ix, fq in enumerate(
                    self.receiver.coefficients[
                        : max(1, min(self.receiver.rcf_taps, 160))
                    ]
                )
            ]
        )
        txt += f"sct.receiver.coefficients:\n {t}\n"
        t = "\n ".join(
            [
                f"{ix+1}: {fq}"
                for ix, fq in enumerate(
                    self.receiver.rx_map[: max(1, min(self.receiver.rx_chan, 16))]
                )
            ]
        )
        txt += f"sct.receiver.rx_map: \n{t}\n"
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
        t = "\n ".join(
            [
                f"{ix+1}: {fq}"
                for ix, fq in enumerate(
                    self.exciter.coefficients[: max(1, min(self.exciter.rcf_taps, 256))]
                )
            ]
        )
        txt += f"sct.exciter.coefficients:\n {t}\n"
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
