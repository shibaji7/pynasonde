import ephem
import numpy as np


class Eclipse(object):
    def __init__(self):
        return

    def calculate_w2naf_shadow(self, d, lat, lon, alt=300.0):
        obsc = eclipse_calc.calculate_obscuration(d, lat, lon, alt)
        return obsc

    def intersection(slef, r0, r1, d, n_s=100):
        A1 = np.zeros([n_s, n_s])
        A2 = np.zeros([n_s, n_s])
        I = np.zeros([n_s, n_s])
        x = np.linspace(-2.0 * r0, 2.0 * r0, num=n_s)
        y = np.linspace(-2.0 * r0, 2.0 * r0, num=n_s)
        xx, yy = np.meshgrid(x, y)
        A1[np.sqrt((xx + d) ** 2.0 + yy**2.0) < r0] = 1.0
        n_sun = np.sum(A1)
        A2[np.sqrt(xx**2.0 + yy**2.0) < r1] = 1.0
        S = A1 + A2
        I[S > 1] = 1.0
        eclipse = np.sum(I) / n_sun
        return eclipse

    def create_eclipse_shadow(self, d, lat, lon, alt):
        obs = ephem.Observer()
        t0 = ephem.date(
            (
                d.year,
                d.month,
                d.day,
                d.hour,
                d.minute,
                d.second,
            )
        )
        obs.lon, obs.lat = "%1.2f" % (lon), "%1.2f" % (lat)  # ESR
        obs.elevation = alt
        obs.date = t0
        sun, moon = ephem.Sun(), ephem.Moon()
        sun.compute(obs)
        moon.compute(obs)
        r_sun = (sun.size / 2.0) / 3600.0
        r_moon = (moon.size / 2.0) / 3600.0
        s = np.degrees(ephem.separation((sun.az, sun.alt), (moon.az, moon.alt)))
        percent_eclipse = 0.0

        if s < (r_moon + r_sun):
            if s < 1e-3:
                percent_eclipse = 1.0
            else:
                percent_eclipse = self.intersection(r_moon, r_sun, s, n_s=100)
        if np.degrees(sun.alt) <= r_sun:
            if np.degrees(sun.alt) <= -r_sun:
                percent_eclipse = 2
            else:
                percent_eclipse = 1.0 - (
                    (np.degrees(sun.alt) + r_sun) / (2.0 * r_sun)
                ) * (1.0 - percent_eclipse)
        return percent_eclipse


def create_eclipse_path_local(dates, lat, lon, alt=300):
    from tqdm import tqdm

    e = Eclipse()
    p = np.nan * np.zeros((len(dates)))
    for i, d in enumerate(tqdm(dates)):
        p[i] = e.create_eclipse_shadow(d, lat, lon, alt)
    return p
