## SCT datatypes

Dataclasses modeling VIPIR SCT header-format header block.

::: pynasonde.vipir.riq.datatypes.sct
    handler: python
    options:
        show_root_heading: true
        show_source: false
        members:
            - read_dtype

::: pynasonde.vipir.riq.datatypes.sct.StationType
    handler: python
    options:
        show_root_heading: true
        show_source: true
        members:
            - read_station_from_file_pointer
            - clean

::: pynasonde.vipir.riq.datatypes.sct.TimingType
    handler: python
    options:
        show_root_heading: true
        show_source: true
        members:
            - read_timing_from_file_pointer
            - clean

::: pynasonde.vipir.riq.datatypes.sct.FrequencyType
    handler: python
    options:
        show_root_heading: true
        show_source: true
        members:
            - read_frequency_from_file_pointer
            - clean

::: pynasonde.vipir.riq.datatypes.sct.RecieverType
    handler: python
    options:
        show_root_heading: true
        show_source: true
        members:
            - read_reciever_from_file_pointer

::: pynasonde.vipir.riq.datatypes.sct.ExciterType
    handler: python
    options:
        show_root_heading: true
        show_source: true
        members:
            - read_exciter_from_file_pointer
            - clean

::: pynasonde.vipir.riq.datatypes.sct.MonitorType
    handler: python
    options:
        show_root_heading: true
        show_source: true
        members:
            - read_monitor_from_file_pointer

::: pynasonde.vipir.riq.datatypes.sct.SctType
    handler: python
    options:
        show_root_heading: true
        show_source: true
        members:
            - read_sct_from_file_pointer
            - fix_SCT_strings
            - dump_sct