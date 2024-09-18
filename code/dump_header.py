from pynasonde.riq.headers.pct import PctType
from pynasonde.riq.headers.sct import SctType

fname = "WI937_2013169113403.RIQ"
x, y = SctType(), PctType()
x.read_sct(fname)
y.load_sct(x)
y.read_pct(fname)
y.dump_pct()
