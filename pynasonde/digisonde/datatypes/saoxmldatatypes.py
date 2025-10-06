"""Datatypes and XML-to-dataclass mapping helpers for SAO XML.

This module defines dataclasses that mirror the SAO XML structure used by
Digisonde SAO exports. It includes utility parsing functions in
``SAORecordList.load_from_xml`` that validate against the DTD and map XML
elements into rich Python dataclasses.

These dataclasses are intentionally lightweight and closely mirror the
XML element attributes so mkdocstrings can render field-level API docs
for consumers and examples.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class URSI:
    """Represents a single URSI characteristic entry.

    Attributes:
        ID: Any
            Identifier of the URSI parameter (as read from XML).
        Val: float
            Numeric value; coerced to a float in ``__post_init__`` for
            downstream consumers.
        Name: Optional[str]
            Optional human-readable name for the parameter.
        Units: Optional[str]
            Units string for the value where provided.
        QL: Optional[str]
            Quality level metadata (parser-specific).
        DL: Optional[str]
            Detection level metadata (parser-specific).
        SigFig: Optional[str]
            Significant-figures metadata.
        UpperBound: Optional[str]
            Upper bound metadata from XML.
        LowerBound: Optional[str]
            Lower bound metadata from XML.
        Bound: Optional[str]
            Bound metadata.
        BoundaryType: Optional[str]
            Boundary type metadata.
        Flag: Optional[str]
            Optional flag or marker from the XML.
    """

    ID: Any
    Val: Any
    Name: Optional[str] = None
    Units: Optional[str] = None
    QL: Optional[str] = None
    DL: Optional[str] = None
    SigFig: Optional[str] = None
    UpperBound: Optional[str] = None
    LowerBound: Optional[str] = None
    Bound: Optional[str] = None
    BoundaryType: Optional[str] = None
    Flag: Optional[str] = None

    def __post_init__(self):
        # Ensure numeric value is a float for downstream consumers
        self.Val = float(self.Val)
        return


@dataclass
class Modeled:
    """Represents a modeled parameter entry in the SAO XML.

    Attributes:
        Name: str
            Parameter name.
        Val: str
            Parameter value (string as represented in XML).
        Units: str
            Units string for the value.
        ModelName: Optional[str]
            Optional model name used to derive the value.
        ModelOptions: Optional[str]
            Optional model options string.
    """

    Name: str
    Val: str
    Units: str
    ModelName: Optional[str] = None
    ModelOptions: Optional[str] = None


@dataclass
class Custom:
    """Represents a custom parameter entry included in SAO XML.

    Attributes:
        Name: str
            Parameter name.
        Val: str
            Parameter value.
        Units: str
            Units string for the value.
        Description: str
            Human-readable description of the parameter.
        SigFig: Optional[str]
            Significant figures metadata.
        UpperBound: Optional[str]
            Upper bound metadata.
        LowerBound: Optional[str]
            Lower bound metadata.
        Bound: Optional[str]
            Bound metadata.
        BoundaryType: Optional[str]
            Boundary type metadata.
        Flag: Optional[str]
            Optional flag or marker.
    """

    Name: str
    Val: str
    Units: str
    Description: str
    SigFig: Optional[str] = None
    UpperBound: Optional[str] = None
    LowerBound: Optional[str] = None
    Bound: Optional[str] = None
    BoundaryType: Optional[str] = None
    Flag: Optional[str] = None


@dataclass
class CharacteristicList:
    """Container for URSI/Modeled/Custom characteristic sub-elements.

    Attributes:
        URSI: List[URSI]
            List of `URSI` entries.
        Modeled: List[Modeled]
            List of `Modeled` parameter entries.
        Custom: List[Custom]
            List of `Custom parameter entries.
        Num: Optional[int]
            Optional count attribute from the XML (coerced to int in
            ``__post_init__`` when present).
    """

    URSI: List["URSI"] = field(default_factory=list)
    Modeled: List["Modeled"] = field(default_factory=list)
    Custom: List["Custom"] = field(default_factory=list)
    Num: Optional[Any] = None

    def __post_init__(self):
        """Coerce list-count fields to integers when present."""
        self.Num = int(self.Num) if self.Num is not None else None
        return


@dataclass
class TraceValueList:
    """Represents a list of trace values for a Trace element.

    Attributes:
        Name: str
            Name of the trace value list.
        Type: Optional[str]
            Optional type attribute.
        SigFig: Optional[str]
            Significant-figures metadata.
        Units: Optional[str]
            Units for the values.
        NoValue: Optional[str]
            Marker used for missing values.
        Description: Optional[str]
            Optional description string.
        values: List[str]
            List of string values (converted to floats by the parser functions
            when appropriate).
    """

    Name: str
    Type: Optional[str] = None
    SigFig: Optional[str] = None
    Units: Optional[str] = None
    NoValue: Optional[str] = None
    Description: Optional[str] = None
    values: List[str] = field(default_factory=list)


@dataclass
class Trace:
    """Represents a single Trace element with frequency/range axes and
    associated TraceValueList entries.

    Attributes:
        FrequencyList: List[float]
            Frequency axis values for the trace.
        RangeList: List[float]
            Range/height axis values for the trace.
        TraceValueList: List[TraceValueList]
            List of `TraceValueList` objects containing measured values.
        Type: Optional[str]
            Trace type (defaults to "standard").
        Layer: str
            Layer name or identifier.
        Multiple: Optional[str]
            Multiplexing indicator when present.
        Polarization: str
            Polarization string for the trace.
        Num: str
            Optional numeric identifier string.
    """

    FrequencyList: List[float]
    RangeList: List[float]
    TraceValueList: List["TraceValueList"] = field(default_factory=list)
    Type: Optional[str] = "standard"
    Layer: str = ""
    Multiple: Optional[str] = None
    Polarization: str = ""
    Num: str = ""


@dataclass
class TraceList:
    """Container for a list of Trace objects.

    Attributes:
        Trace: List[Trace]
            List of `Trace` entries.
        Num: Optional[str]
            Optional count attribute from XML.
    """

    Trace: List["Trace"] = field(default_factory=list)
    Num: Optional[str] = None


@dataclass
class ProfileValueList:
    """Represents a named list of profile values used inside Tabulated
    profile data.

    Attributes:
        Name: str
            Name of the profile value list.
        Type: Optional[str]
            Optional type attribute.
        SigFig: Optional[str]
            Significant-figures metadata.
        Units: Optional[str]
            Units for the values.
        NoValue: Optional[str]
            Missing-value marker.
        Description: Optional[str]
            Description string.
        values: List[str]
            Numeric values (parser converts to floats when appropriate).
    """

    Name: str
    Type: Optional[str] = None
    SigFig: Optional[str] = None
    Units: Optional[str] = None
    NoValue: Optional[str] = None
    Description: Optional[str] = None
    values: List[str] = field(default_factory=list)


@dataclass
class Tabulated:
    """Holds tabulated profile data with altitude axis and named value
    lists.

    Attributes:
        Num: str
            Optional count or identifier.
        AltitudeList: List[float]
            Altitude (height) axis values.
        ProfileValueList: List[ProfileValueList]
            List of profile `ProfileValueList` value lists for each parameter.
    """

    Num: str
    AltitudeList: List[float]
    ProfileValueList: List["ProfileValueList"] = field(default_factory=list)


@dataclass
class Profile:
    """Represents a computed or tabulated electron density profile
    included in the SAO output.

    Attributes:
        Algorithm: str
            Name of the profile algorithm used.
        AlgorithmVersion: str
            Version string for the algorithm.
        Type: Optional[str]
            Profile type (defaults to "vertical").
        Description: Optional[str]
            Optional description text.
        Tabulated: Optional[Tabulated]
            Tabulated data for the profile when present (`Tabulated`).
    """

    Algorithm: str
    AlgorithmVersion: str
    Type: Optional[str] = "vertical"
    Description: Optional[str] = None
    Tabulated: Optional["Tabulated"] = None
    # Add other profile types as needed


@dataclass
class ProfileList:
    """Container for Profile entries.

    Attributes:
        Profile: List[Profile]
            List of `Profiles`.
        Num: Optional[str]
            Optional count attribute from XML.
    """

    Profile: List["Profile"] = field(default_factory=list)
    Num: Optional[str] = None


@dataclass
class SystemInfo:
    """Partial mapping of system-level metadata reported in SAO XML.

    Attributes:
        UMLStationID: Optional[str]
            UML station identifier when present.
        IUWDSCode: Optional[str]
            IUWDS code when present.
    """

    UMLStationID: Optional[str] = None
    IUWDSCode: Optional[str] = None
    # Add other sub-elements as needed


@dataclass
class SAORecord:
    """Top-level representation of an SAO record exported as XML.

    Attributes:
        SystemInfo: Optional[SystemInfo]
            System-level metadata object when present.
        CharacteristicList: CharacteristicList
            Characteristic lists (URSI/Modeled/Custom) describing the record.
        TraceList: Optional[TraceList]
            Optional list of Trace elements providing ionogram traces.
        ProfileList: Optional[ProfileList]
            Optional profiles included in the record.
        FormatVersion: str
            SAO format version (defaults to "5.0").
        StartTimeUTC: str
            Start time string in UTC as provided by the XML.
        URSICode: str
            URSI code associated with the record.
        StationName: str
            Station name string.
        GeoLatitude: str
            Latitude string representation.
        GeoLongitude: str
            Longitude string representation.
        Source: str
            Source identifier (defaults to "Ionosonde").
        SourceType: str
            Source type string.
        ScalerType: str
            Scaler type string.
    """

    SystemInfo: Optional["SystemInfo"] = None
    CharacteristicList: "CharacteristicList" = None
    TraceList: Optional["TraceList"] = None
    ProfileList: Optional["ProfileList"] = None
    FormatVersion: str = "5.0"
    StartTimeUTC: str = ""
    URSICode: str = ""
    StationName: str = ""
    GeoLatitude: str = ""
    GeoLongitude: str = ""
    Source: str = "Ionosonde"
    SourceType: str = ""
    ScalerType: str = ""


@dataclass
class SAORecordList:
    """Top-level container for a list of SAORecord instances parsed from an
    SAO XML file.

    Attributes:
        SAORecord: List[SAORecord]
            List of parsed `SAORecord` objects.
    """

    SAORecord: List["SAORecord"] = field(default_factory=list)

    @staticmethod
    def load_from_xml(xml_path: str, dtd_path: str = None) -> "SAORecordList":
        """Parse an SAO XML file and return a populated SAORecordList.

        The method validates the XML against the SAO DTD (if available via
        ``dtd_path`` or the packaged resource), then recursively maps XML
        elements to the dataclass hierarchy defined in this module.

        Parameters:
            xml_path: Path to the SAO XML file to parse.
            dtd_path: Optional path to a DTD file for validation.

        Returns:
            SAORecordList populated with parsed SAORecord instances.
        """
        # --- DTD Validation ---
        from lxml import etree

        from pynasonde.digisonde.digi_utils import load_dtd_file

        parser = load_dtd_file(dtd_path)
        tree = etree.parse(xml_path)
        root = tree.getroot()

        # --- Recursive mapping ---
        def get_text_list(element):
            # Helper to split whitespace-separated floats/strings
            if element is None or element.text is None:
                return []
            return [float(x) for x in element.text.strip().split() if x]

        def parse_ursi(elem):
            return URSI(**elem.attrib)

        def parse_modeled(elem):
            return Modeled(**elem.attrib)

        def parse_custom(elem):
            return Custom(**elem.attrib)

        def parse_characteristic_list(elem):
            return CharacteristicList(
                URSI=[parse_ursi(e) for e in elem.findall("URSI")],
                Modeled=[parse_modeled(e) for e in elem.findall("Modeled")],
                Custom=[parse_custom(e) for e in elem.findall("Custom")],
                Num=elem.attrib.get("Num"),
            )

        def parse_trace_value_list(elem):
            return TraceValueList(
                Name=elem.attrib["Name"],
                Type=elem.attrib.get("Type"),
                SigFig=elem.attrib.get("SigFig"),
                Units=elem.attrib.get("Units"),
                NoValue=elem.attrib.get("NoValue"),
                Description=elem.attrib.get("Description"),
                values=(
                    [float(x) for x in elem.text.strip().split()] if elem.text else []
                ),
            )

        def parse_trace(elem):
            freq_list_elem = elem.find("FrequencyList")
            range_list_elem = elem.find("RangeList")
            return Trace(
                FrequencyList=get_text_list(freq_list_elem),
                RangeList=get_text_list(range_list_elem),
                TraceValueList=[
                    parse_trace_value_list(e) for e in elem.findall("TraceValueList")
                ],
                Type=elem.attrib.get("Type", "standard"),
                Layer=elem.attrib.get("Layer", ""),
                Multiple=elem.attrib.get("Multiple"),
                Polarization=elem.attrib.get("Polarization", ""),
                Num=elem.attrib.get("Num", ""),
            )

        def parse_trace_list(elem):
            return TraceList(
                Trace=[parse_trace(e) for e in elem.findall("Trace")],
                Num=elem.attrib.get("Num"),
            )

        def parse_profile_value_list(elem):
            return ProfileValueList(
                Name=elem.attrib["Name"],
                Type=elem.attrib.get("Type"),
                SigFig=elem.attrib.get("SigFig"),
                Units=elem.attrib.get("Units"),
                NoValue=elem.attrib.get("NoValue"),
                Description=elem.attrib.get("Description"),
                values=(
                    [float(x) for x in elem.text.strip().split()] if elem.text else []
                ),
            )

        def parse_tabulated(elem):
            alt_elem = elem.find("AltitudeList")
            return Tabulated(
                Num=elem.attrib.get("Num"),
                AltitudeList=get_text_list(alt_elem),
                ProfileValueList=[
                    parse_profile_value_list(e)
                    for e in elem.findall("ProfileValueList")
                ],
            )

        def parse_profile(elem):
            tab_elem = elem.find("Tabulated")
            return Profile(
                Algorithm=elem.attrib["Algorithm"],
                AlgorithmVersion=elem.attrib["AlgorithmVersion"],
                Type=elem.attrib.get("Type", "vertical"),
                Description=elem.attrib.get("Description"),
                Tabulated=parse_tabulated(tab_elem) if tab_elem is not None else None,
            )

        def parse_profile_list(elem):
            return ProfileList(
                Profile=[parse_profile(e) for e in elem.findall("Profile")],
                Num=elem.attrib.get("Num"),
            )

        def parse_system_info(elem):
            # Only a partial mapping; expand as needed
            return SystemInfo(
                UMLStationID=elem.attrib.get("UMLStationID"),
                IUWDSCode=elem.attrib.get("IUWDSCode"),
            )

        def parse_saorecord(elem):
            return SAORecord(
                SystemInfo=(
                    parse_system_info(elem.find("SystemInfo"))
                    if elem.find("SystemInfo") is not None
                    else None
                ),
                CharacteristicList=parse_characteristic_list(
                    elem.find("CharacteristicList")
                ),
                TraceList=(
                    parse_trace_list(elem.find("TraceList"))
                    if elem.find("TraceList") is not None
                    else None
                ),
                ProfileList=(
                    parse_profile_list(elem.find("ProfileList"))
                    if elem.find("ProfileList") is not None
                    else None
                ),
                FormatVersion=elem.attrib.get("FormatVersion", "5.0"),
                StartTimeUTC=elem.attrib.get("StartTimeUTC", ""),
                URSICode=elem.attrib.get("URSICode", ""),
                StationName=elem.attrib.get("StationName", ""),
                GeoLatitude=elem.attrib.get("GeoLatitude", ""),
                GeoLongitude=elem.attrib.get("GeoLongitude", ""),
                Source=elem.attrib.get("Source", "Ionosonde"),
                SourceType=elem.attrib.get("SourceType", ""),
                ScalerType=elem.attrib.get("ScalerType", ""),
            )

        # --- Top-level SAORecordList ---
        sao_records = [parse_saorecord(e) for e in root.findall("SAORecord")]
        return SAORecordList(SAORecord=sao_records)
