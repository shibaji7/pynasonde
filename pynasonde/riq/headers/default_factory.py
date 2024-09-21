from typing import List, Tuple

SCT_default_factory: List[Tuple] = [
    ("magic", "int32"),  # magic number
    (
        "sounding_table_size",
        "int32",
    ),  # bytes in sounder configuration structure
    ("pulse_table_size", "int32"),  # bytes in pulse configuration structure
    ("raw_data_size", "int32"),  # bytes in raw data block (one PRI)
    ("struct_version", "float32"),  # Format Version Number
    ("start_year", "int32"),  # Start Year
    ("start_daynumber", "int32"),  # Start Day Number
    ("start_month", "int32"),  # Start Month
    ("start_day", "int32"),  # Start Day
    ("start_hour", "int32"),  # Start Hour
    ("start_minute", "int32"),  # Start Minute
    ("start_second", "int32"),  # Start Second
    ("start_epoch", "int32"),  # Start epoch time
    ("readme", "S4", (32,)),  # Operator comment (32x4 byte string)
    ("decimation_method", "int32"),  # Decimation method
    ("decimation_threshold", "float32"),  # Decimation threshold
    ("user", "S4", (32,)),  # User-defined (32x4 byte string)
]

Station_default_factory: List[Tuple] = [
    ("file_id", "S4", (16,)),  # "uint8" [4 16]
    ("ursi_id", "S4", (2,)),  # "uint8" [4 2]
    ("rx_name", "S4", (8,)),  # "uint8" [4 8]
    ("rx_latitude", "float32"),  # "single" [1 1]
    ("rx_longitude", "float32"),  # "single" [1 1]
    ("rx_altitude", "float32"),  # "single" [1 1]
    ("rx_count", "int32"),  # "int32" [1 1]
    ("rx_antenna_type", "S4", (32, 8)),  # "uint8" [8 32]
    ("rx_position", "float32", (32, 3)),  # "single" [1 96]
    ("rx_direction", "float32", (32, 3)),  # "single" [1 96]
    ("rx_height", "float32", (32,)),  # "single" [1 32]
    ("rx_cable_length", "float32", (32,)),  # "single" [1 32]
    ("frontend_atten", "float32"),  # "single" [1 1]
    ("tx_name", "S4", (8,)),  # "uint8" [4 8]
    ("tx_latitude", "float32"),  # "single" [1 1]
    ("tx_longitude", "float32"),  # "single" [1 1]
    ("tx_altitude", "float32"),  # "single" [1 1]
    ("tx_antenna_type", "S4", (8,)),  # "uint8" [4 8]
    ("tx_vector", "float32", (3,)),  # "single" [1 3]
    ("tx_height", "float32"),  # "single" [1 1]
    ("tx_cable_length", "float32"),  # "single" [1 1]
    ("drive_band_count", "int32"),  # "int32" [1 1]
    ("drive_band_bounds", "float32", (64, 2)),  # "single" [1 128]
    ("drive_band_atten", "float32", (64,)),  # "single" [1 64]
    ("rf_control", "int32"),  # "int32" [1 1]
    ("ref_type", "S4", (8,)),  # "uint8" [4 8]
    ("clock_type", "S4", (8,)),  # "uint8" [4 8]
    ("user", "S4", (32,)),  # "uint8" [4 32]
]


Timing_default_factory: List[Tuple] = [
    ("file_id", "S4", (16,)),  # "uint8" [4 16]
    ("pri", "float32"),  # "single" [1 1]
    ("pri_count", "int32"),  # "int32" [1 1]
    ("ionogram_count", "int32"),  # "int32" [1 1]
    ("holdoff", "float32"),  # "single" [1 1]
    ("range_gate_offset", "float32"),  # "single" [1 1]
    ("gate_count", "int32"),  # "int32" [1 1]
    ("gate_start", "float32"),  # "single" [1 1]
    ("gate_end", "float32"),  # "single" [1 1]
    ("gate_step", "float32"),  # "single" [1 1]
    ("data_start", "float32"),  # "single" [1 1]
    ("data_width", "float32"),  # "single" [1 1]
    ("data_baud_count", "int32"),  # "int32" [1 1]
    ("data_wave_file", "S4", (16,)),  # "uint8" [4 16]
    ("data_baud", "float32", (1024, 2)),  # "single" [1024 2]
    ("data_pairs", "int32"),  # "int32" [1 1]
    ("cal_start", "float32"),  # "single" [1 1]
    ("cal_width", "float32"),  # "single" [1 1]
    ("cal_baud_count", "int32"),  # "int32" [1 1]
    ("cal_wave_file", "S4", (16,)),  # "uint8" [4 16]
    ("cal_baud", "float32", (1024, 2)),  # "single" [1 2048]
    ("cal_pairs", "int32"),  # "int32" [1 1]
    ("user", "S4", (32,)),  # "uint8" [4 32]
]


Frequency_default_factory: List[Tuple] = [
    ("file_id", "S4", (16,)),  # "uint8" [4 16]
    ("base_start", "float32"),  # "single" [1 1]
    ("base_end", "float32"),  # "single" [1 1]
    ("base_steps", "int32"),  # "int32" [1 1]
    ("tune_type", "int32"),  # "int32" [1 1]
    ("base_table", "float32", (8192,)),  # "single" [1 8192]
    ("linear_step", "float32"),  # "single" [1 1]
    ("log_step", "float32"),  # "single" [1 1]
    ("freq_table_id", "S4", (16,)),  # "uint8" [4 16]
    ("tune_steps", "int32"),  # "int32" [1 1]
    ("pulse_count", "int32"),  # "int32" [1 1]
    ("pulse_pattern", "int32", (256,)),  # "int32" [1 256]
    ("pulse_offset", "float32"),  # "single" [1 1]
    ("ramp_steps", "int32"),  # "int32" [1 1]
    ("ramp_repeats", "int32"),  # "int32" [1 1]
    ("drive_table", "float32", (8192,)),  # "single" [1 8192]
    ("user", "S4", (32,)),  # "uint8" [4 32]
]

Reciever_default_factory: List[Tuple] = [
    ("file_id", "S4", (16,)),  # "uint8" [4 16]
    ("rx_chan", "int32"),  # "int32" [1 1]
    ("rx_map", "int32", (16)),  # "int32" [1 16]
    ("word_format", "int32"),  # "int32" [1 1]
    ("cic2_dec", "int32"),  # "int32" [1 1]
    ("cic2_interp", "int32"),  # "int32" [1 1]
    ("cic2_scale", "int32"),  # "int32" [1 1]
    ("cic5_dec", "int32"),  # "int32" [1 1]
    ("cic5_scale", "int32"),  # "int32" [1 1]
    ("rcf_type", "S4", (8,)),  # "uint8" [4 8]
    ("rcf_dec", "int32"),  # "int32" [1 1]
    ("rcf_taps", "int32"),  # "int32" [1 1]
    ("coefficients", "int32", (160,)),  # "int32" [1 160]
    ("analog_delay", "float32"),  # "single" [1 1]
    ("user", "S4", (32,)),  # "uint8" [4 32]
]

Exciter_default_factory: List[Tuple] = [
    ("file_id", "S4", (16,)),  # "uint8" [4 16]
    ("cic_scale", "int32"),  # "int32" [1 1]
    ("cic2_dec", "int32"),  # "int32" [1 16]
    ("cic2_interp", "int32"),  # "int32" [1 1]
    ("cic5_interp", "int32"),  # "int32" [1 1]
    ("rcf_type", "S4", (8,)),  # "uint8" [4 8]
    ("rcf_taps", "int32"),  # "int32" [1 1]
    ("rcf_taps_phase", "int32"),  # "int32" [1 1]
    ("coefficients", "int32", (256,)),  # "int32" [1 256]
    ("analog_delay", "float32"),  # "single" [1 1]
    ("user", "S4", (32,)),  # "uint8" [4 32]
]

Monitor_default_factory: List[Tuple] = [
    ("balun_currents", "int32"),
    ("balun_status", "int32"),
    ("front_end_status", "int32"),
    ("receiver_status", "int32"),
    ("exciter_status", "int32"),
    ("user", "S4", (128,)),
]

PCT_default_factory: List[Tuple] = [
    ("record_id", "int32"),
    ("invalid", "int32", (4,)),
    ("base_id", "int32"),
    ("pulse_id", "int32"),
    ("ramp_id", "int32"),
    ("repeat_id", "int32"),
    ("loop_id", "int32"),
    ("frequency", "float32"),
    ("nco_tune_word", "int32"),
    ("drive_attenuation", "float32"),
    ("pa_flags", "int32"),
    ("pa_forward_power", "float32"),
    ("pa_reflected_power", "float32"),
    ("pa_vswr", "float32"),
    ("pa_temperature", "float32"),
    ("proc_range_count", "int32"),
    ("proc_noise_level", "float32"),
    ("user", "S4", (16,)),
]
