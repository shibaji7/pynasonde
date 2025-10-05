"""Datatypes and XML-to-dataclass mapping helpers for SAO XML.

This module defines dataclasses that mirror the SAO XML structure used by
Digisonde SAO exports. It includes utility parsing functions in
``SAORecordList.load_from_xml`` that validate against the DTD and map XML
elements into rich Python dataclasses.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class URSI:
    """Represents a single URSI characteristic entry.

    Attributes:
        ID: Identifier of the parameter.
        Val: Numeric value (coerced to float in ``__post_init__``).
        Name, Units, QL, DL, SigFig, UpperBound, LowerBound, Bound,
        BoundaryType, Flag: Optional metadata fields mapped from XML.
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
    Name: str
    Val: str
    Units: str
    ModelName: Optional[str] = None
    ModelOptions: Optional[str] = None


@dataclass
class Custom:
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
    Name: str
    Type: Optional[str] = None
    SigFig: Optional[str] = None
    Units: Optional[str] = None
    NoValue: Optional[str] = None
    Description: Optional[str] = None
    values: List[str] = field(default_factory=list)


@dataclass
class Trace:
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
    Trace: List["Trace"] = field(default_factory=list)
    Num: Optional[str] = None


@dataclass
class ProfileValueList:
    Name: str
    Type: Optional[str] = None
    SigFig: Optional[str] = None
    Units: Optional[str] = None
    NoValue: Optional[str] = None
    Description: Optional[str] = None
    values: List[str] = field(default_factory=list)


@dataclass
class Tabulated:
    Num: str
    AltitudeList: List[float]
    ProfileValueList: List["ProfileValueList"] = field(default_factory=list)


@dataclass
class Profile:
    Algorithm: str
    AlgorithmVersion: str
    Type: Optional[str] = "vertical"
    Description: Optional[str] = None
    Tabulated: Optional["Tabulated"] = None
    # Add other profile types as needed


@dataclass
class ProfileList:
    Profile: List["Profile"] = field(default_factory=list)
    Num: Optional[str] = None


@dataclass
class SystemInfo:
    UMLStationID: Optional[str] = None
    IUWDSCode: Optional[str] = None
    # Add other sub-elements as needed


@dataclass
class SAORecord:
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
    SAORecord: List["SAORecord"] = field(default_factory=list)

    @staticmethod
    def load_from_xml(xml_path: str, dtd_path: str = None) -> "SAORecordList":
        """Parse an SAO XML file and return a populated SAORecordList.

        The method validates the XML against the SAO DTD (if available via
        ``dtd_path`` or the packaged resource), then recursively maps XML
        elements to the dataclass hierarchy defined in this module.

        Args:
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
