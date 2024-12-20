import datetime as dt

from pynasonde.model.point import Point

if __name__ == "__main__":
    p = Point(dt.datetime(2017, 5, 27, 16), 42.6233, -71.4882)
    p._load_profile_()
