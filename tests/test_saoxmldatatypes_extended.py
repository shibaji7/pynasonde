"""Extended tests for pynasonde.digisonde.datatypes.saoxmldatatypes.

Covers all dataclass constructors and __post_init__ conversions without
requiring an XML file or DTD: URSI, Modeled, Custom, CharacteristicList,
TraceValueList, Trace, TraceList, ProfileValueList, Tabulated, Profile,
ProfileList, SystemInfo, SAORecord, SAORecordList.
"""

import pytest

from pynasonde.digisonde.datatypes.saoxmldatatypes import (
    CharacteristicList,
    Custom,
    Modeled,
    Profile,
    ProfileList,
    ProfileValueList,
    SAORecord,
    SAORecordList,
    SystemInfo,
    Tabulated,
    Trace,
    TraceList,
    TraceValueList,
    URSI,
)


# ---------------------------------------------------------------------------
# URSI
# ---------------------------------------------------------------------------

class TestURSI:
    def test_val_coerced_to_float_from_string(self):
        u = URSI(ID="foF2", Val="5.3")
        assert u.Val == pytest.approx(5.3)
        assert isinstance(u.Val, float)

    def test_val_already_float(self):
        u = URSI(ID="hmF2", Val=300.0)
        assert u.Val == pytest.approx(300.0)

    def test_val_zero(self):
        u = URSI(ID="fmin", Val="0")
        assert u.Val == pytest.approx(0.0)

    def test_optional_fields_default_none(self):
        u = URSI(ID="foE", Val="3.0")
        assert u.Name is None
        assert u.Units is None
        assert u.QL is None
        assert u.DL is None
        assert u.SigFig is None
        assert u.UpperBound is None
        assert u.LowerBound is None
        assert u.Bound is None
        assert u.BoundaryType is None
        assert u.Flag is None

    def test_all_fields_set(self):
        u = URSI(
            ID="foF2",
            Val="7.2",
            Name="foF2",
            Units="MHz",
            QL="5",
            DL="4",
            SigFig="2",
            UpperBound="10.0",
            LowerBound="1.0",
            Bound="1.5",
            BoundaryType="U",
            Flag="A",
        )
        assert u.ID == "foF2"
        assert u.Name == "foF2"
        assert u.Units == "MHz"
        assert u.QL == "5"
        assert u.Val == pytest.approx(7.2)


# ---------------------------------------------------------------------------
# Modeled
# ---------------------------------------------------------------------------

class TestModeled:
    def test_basic_instantiation(self):
        m = Modeled(Name="hmF2", Val="310.5", Units="km")
        assert m.Name == "hmF2"
        assert m.Val == "310.5"
        assert m.Units == "km"

    def test_optional_fields_none(self):
        m = Modeled(Name="foF2", Val="7.0", Units="MHz")
        assert m.ModelName is None
        assert m.ModelOptions is None

    def test_with_model_fields(self):
        m = Modeled(Name="foF2", Val="7.0", Units="MHz",
                    ModelName="IRI-2016", ModelOptions="default")
        assert m.ModelName == "IRI-2016"
        assert m.ModelOptions == "default"


# ---------------------------------------------------------------------------
# Custom
# ---------------------------------------------------------------------------

class TestCustom:
    def test_basic_instantiation(self):
        c = Custom(Name="TEC", Val="10.5", Units="TECU",
                   Description="Total electron content")
        assert c.Name == "TEC"
        assert c.Val == "10.5"
        assert c.Units == "TECU"
        assert c.Description == "Total electron content"

    def test_optional_fields_none(self):
        c = Custom(Name="x", Val="1", Units="u", Description="d")
        assert c.SigFig is None
        assert c.UpperBound is None
        assert c.LowerBound is None
        assert c.Bound is None
        assert c.BoundaryType is None
        assert c.Flag is None


# ---------------------------------------------------------------------------
# CharacteristicList
# ---------------------------------------------------------------------------

class TestCharacteristicList:
    def test_num_coerced_to_int(self):
        cl = CharacteristicList(Num="3")
        assert cl.Num == 3
        assert isinstance(cl.Num, int)

    def test_num_none_stays_none(self):
        cl = CharacteristicList(Num=None)
        assert cl.Num is None

    def test_empty_lists_default(self):
        cl = CharacteristicList()
        assert cl.URSI == []
        assert cl.Modeled == []
        assert cl.Custom == []

    def test_ursi_list_stored(self):
        u1 = URSI(ID="foF2", Val="5.0")
        u2 = URSI(ID="hmF2", Val="300.0")
        cl = CharacteristicList(URSI=[u1, u2], Num="2")
        assert len(cl.URSI) == 2
        assert cl.URSI[0].ID == "foF2"
        assert cl.Num == 2

    def test_modeled_list_stored(self):
        m = Modeled(Name="foF2", Val="5.1", Units="MHz")
        cl = CharacteristicList(Modeled=[m])
        assert len(cl.Modeled) == 1
        assert cl.Modeled[0].Name == "foF2"

    def test_custom_list_stored(self):
        c = Custom(Name="TEC", Val="12.0", Units="TECU", Description="TEC")
        cl = CharacteristicList(Custom=[c])
        assert len(cl.Custom) == 1


# ---------------------------------------------------------------------------
# TraceValueList
# ---------------------------------------------------------------------------

class TestTraceValueList:
    def test_basic_with_values(self):
        tvl = TraceValueList(Name="Amplitude", values=[1.0, 2.0, 3.0])
        assert tvl.Name == "Amplitude"
        assert tvl.values == [1.0, 2.0, 3.0]

    def test_optional_fields_default_none(self):
        tvl = TraceValueList(Name="Phase")
        assert tvl.Type is None
        assert tvl.SigFig is None
        assert tvl.Units is None
        assert tvl.NoValue is None
        assert tvl.Description is None

    def test_empty_values_default(self):
        tvl = TraceValueList(Name="Test")
        assert tvl.values == []

    def test_all_fields(self):
        tvl = TraceValueList(
            Name="Amplitude", Type="float", SigFig="3",
            Units="dB", NoValue="-1.0", Description="Signal amplitude",
            values=[1.1, 2.2],
        )
        assert tvl.Type == "float"
        assert tvl.SigFig == "3"
        assert tvl.NoValue == "-1.0"


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------

class TestTrace:
    def test_basic_instantiation(self):
        t = Trace(FrequencyList=[2.0, 3.0, 4.0], RangeList=[100.0, 150.0, 200.0])
        assert len(t.FrequencyList) == 3
        assert len(t.RangeList) == 3

    def test_default_type_standard(self):
        t = Trace(FrequencyList=[], RangeList=[])
        assert t.Type == "standard"

    def test_optional_fields(self):
        t = Trace(FrequencyList=[3.0], RangeList=[200.0],
                  Layer="F2", Polarization="O", Num="1")
        assert t.Layer == "F2"
        assert t.Polarization == "O"
        assert t.Num == "1"

    def test_trace_value_list(self):
        tvl = TraceValueList(Name="Amp", values=[10.0, 20.0])
        t = Trace(FrequencyList=[3.0], RangeList=[200.0],
                  TraceValueList=[tvl])
        assert len(t.TraceValueList) == 1
        assert t.TraceValueList[0].Name == "Amp"


# ---------------------------------------------------------------------------
# TraceList
# ---------------------------------------------------------------------------

class TestTraceList:
    def test_empty_list(self):
        tl = TraceList()
        assert tl.Trace == []
        assert tl.Num is None

    def test_with_traces(self):
        t = Trace(FrequencyList=[2.0], RangeList=[100.0])
        tl = TraceList(Trace=[t], Num="1")
        assert len(tl.Trace) == 1
        assert tl.Num == "1"


# ---------------------------------------------------------------------------
# ProfileValueList
# ---------------------------------------------------------------------------

class TestProfileValueList:
    def test_basic_instantiation(self):
        pvl = ProfileValueList(Name="PlasmaFrequency", values=[3.0, 4.0, 5.0])
        assert pvl.Name == "PlasmaFrequency"
        assert pvl.values == [3.0, 4.0, 5.0]

    def test_defaults_none(self):
        pvl = ProfileValueList(Name="ED")
        assert pvl.Type is None
        assert pvl.SigFig is None
        assert pvl.Units is None
        assert pvl.NoValue is None
        assert pvl.Description is None
        assert pvl.values == []


# ---------------------------------------------------------------------------
# Tabulated
# ---------------------------------------------------------------------------

class TestTabulated:
    def test_basic_instantiation(self):
        tab = Tabulated(
            Num="10",
            AltitudeList=[100.0, 150.0, 200.0],
        )
        assert tab.Num == "10"
        assert tab.AltitudeList == [100.0, 150.0, 200.0]
        assert tab.ProfileValueList == []

    def test_with_profile_value_lists(self):
        pvl = ProfileValueList(Name="PF", values=[3.0, 4.0])
        tab = Tabulated(Num="2", AltitudeList=[100.0, 150.0],
                        ProfileValueList=[pvl])
        assert len(tab.ProfileValueList) == 1
        assert tab.ProfileValueList[0].Name == "PF"


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

class TestProfile:
    def test_basic_instantiation(self):
        p = Profile(Algorithm="NHPC", AlgorithmVersion="3.0")
        assert p.Algorithm == "NHPC"
        assert p.AlgorithmVersion == "3.0"

    def test_default_type_vertical(self):
        p = Profile(Algorithm="NHPC", AlgorithmVersion="3.0")
        assert p.Type == "vertical"

    def test_optional_fields(self):
        tab = Tabulated(Num="5", AltitudeList=[100.0, 200.0])
        p = Profile(Algorithm="NHPC", AlgorithmVersion="3.0",
                    Description="Electron density profile",
                    Tabulated=tab)
        assert p.Description == "Electron density profile"
        assert p.Tabulated is tab
        assert p.Tabulated.Num == "5"


# ---------------------------------------------------------------------------
# ProfileList
# ---------------------------------------------------------------------------

class TestProfileList:
    def test_empty_list(self):
        pl = ProfileList()
        assert pl.Profile == []
        assert pl.Num is None

    def test_with_profiles(self):
        p = Profile(Algorithm="NHPC", AlgorithmVersion="3.0")
        pl = ProfileList(Profile=[p], Num="1")
        assert len(pl.Profile) == 1
        assert pl.Num == "1"


# ---------------------------------------------------------------------------
# SystemInfo
# ---------------------------------------------------------------------------

class TestSystemInfo:
    def test_defaults_none(self):
        si = SystemInfo()
        assert si.UMLStationID is None
        assert si.IUWDSCode is None

    def test_with_values(self):
        si = SystemInfo(UMLStationID="KR835", IUWDSCode="KR835")
        assert si.UMLStationID == "KR835"
        assert si.IUWDSCode == "KR835"


# ---------------------------------------------------------------------------
# SAORecord
# ---------------------------------------------------------------------------

class TestSAORecord:
    def test_minimal_instantiation(self):
        cl = CharacteristicList()
        rec = SAORecord(CharacteristicList=cl)
        assert rec.CharacteristicList is cl

    def test_default_string_fields(self):
        rec = SAORecord(CharacteristicList=CharacteristicList())
        assert rec.FormatVersion == "5.0"
        assert rec.Source == "Ionosonde"
        assert rec.StartTimeUTC == ""
        assert rec.URSICode == ""

    def test_all_top_level_fields(self):
        si = SystemInfo(UMLStationID="KR835")
        cl = CharacteristicList(URSI=[URSI(ID="foF2", Val="6.0")])
        tl = TraceList()
        pl = ProfileList()
        rec = SAORecord(
            SystemInfo=si,
            CharacteristicList=cl,
            TraceList=tl,
            ProfileList=pl,
            FormatVersion="5.0",
            StartTimeUTC="2024-04-09T00:09:13Z",
            URSICode="KR835",
            StationName="Kazan",
            GeoLatitude="55.8",
            GeoLongitude="48.8",
            Source="Ionosonde",
            SourceType="DPS4D",
            ScalerType="autoscaler",
        )
        assert rec.URSICode == "KR835"
        assert rec.StationName == "Kazan"
        assert rec.GeoLatitude == "55.8"
        assert rec.CharacteristicList.URSI[0].Val == pytest.approx(6.0)

    def test_optional_fields_none_by_default(self):
        rec = SAORecord(CharacteristicList=CharacteristicList())
        assert rec.SystemInfo is None
        assert rec.TraceList is None
        assert rec.ProfileList is None


# ---------------------------------------------------------------------------
# SAORecordList
# ---------------------------------------------------------------------------

class TestSAORecordList:
    def test_empty_list(self):
        srl = SAORecordList()
        assert srl.SAORecord == []

    def test_with_records(self):
        r1 = SAORecord(CharacteristicList=CharacteristicList())
        r2 = SAORecord(CharacteristicList=CharacteristicList(),
                       URSICode="KR835")
        srl = SAORecordList(SAORecord=[r1, r2])
        assert len(srl.SAORecord) == 2
        assert srl.SAORecord[1].URSICode == "KR835"
