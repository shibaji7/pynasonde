import datetime as dt

import numpy as np

from pynasonde.model.point import Point

if __name__ == "__main__":
    p = Point(dt.datetime(2017, 5, 27, 16), 42.6233, -71.4882, np.arange(50, 500))
    p._load_profile_()
    p.calculate_collision_freqs()
    p.calculate_absorptions()
    print(p.H.shape, p.edens.shape)
