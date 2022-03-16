import pandas as pd
import datetime
import time
from skymodel import *
from sunpath import solar

# def getWeatherSolarData(latitude, longitude, weatherData):
#     weatherData['temperature_3'] = weatherData.temperature.shift(3)
#     weatherData.fillna(method='ffill', inplace=True)

#     Ih = [] # global solar radiation on the horizontal surface in W/m2
#     In = [] # direct normal radiation
#     Id = [] # diffuse radiation
#     Kt = [] # clearness index
#     Kn = [] # direct beam transmittance in the form of Gompertz function

#     for i in range(len(weatherData)):
#         when = pd.to_datetime(weatherData.time[i]).timestamp()
#         when = time.gmtime(int(when))
#         temperature = weatherData.temperature[i]
#         temperature_3 = weatherData.temperature_3[i]
#         humidity = weatherData.humidity[i]
#         cloudCover = weatherData.cloudCover[i]
#         windSpeed = weatherData.windSpeed[i]

#         # Zhang-Huang Solar Model
#         output = solar.get_ZhangHuangSolarModel(when, latitude, longitude, temperature, temperature_3, humidity, cloudCover, windSpeed)
#         Ih.append(output[0])
#         In.append(output[1])
#         Id.append(output[2])
#         Kt.append(output[3])
#         Kn.append(output[4])

#     weatherData['Ih'] = Ih
#     weatherData['In'] = In
#     weatherData['Id'] = Id
#     weatherData['Kt'] = Kt
#     weatherData['Kn'] = Kn
#     return weatherData

# weatherData = pd.read_csv('data/300SW/300SW_2021-11-30_2022-02-10_Weather.csv', index_col=[18], parse_dates=True)
# latitude = 26.1199551
# longitude = -80.1468304
# weatherData = getWeatherSolarData(latitude, longitude, weatherData)
# weatherData.to_csv('data/300SW/300SW_weather_solar.csv')

# testing
if __name__ == "__main__":
    # use the current time and my time zone
    my_fmt = "%Y-%m-%d %H:%M:%S"       # datestamp format
    my_tz = -5                    # timezone (GMT/UTC) offset
    my_lat = 26.1199551                            # lat (N positive)
    my_lon = -80.1468304                     # lon (E positive)

    # read weather data
    weatherData = pd.read_csv(
        'data/300SW/300SW_2021-11-30_2022-02-10_Weather.csv', index_col=[18], parse_dates=True)

    weatherData['temperature_3'] = weatherData.temperature.shift(3)
    weatherData.fillna(method='ffill', inplace=True)
    weatherData['doys'] = [x.timetuple().tm_yday for x in weatherData.index]

    Ih = []  # global solar radiation on the horizontal surface in W/m2
    In = []  # direct normal radiation
    Id = []  # diffuse radiation
    solar_altitude = []  # solar altitude

    for i in range(len(weatherData)):
        my_datestamp = weatherData.time[i][:-6]
        dry_bulb_present = weatherData.temperature[i]
        dry_bulb_t3_hrs = weatherData.temperature_3[i]
        relative_humidity = weatherData.humidity[i]
        cloud_cover = weatherData.cloudCover[i] * 10
        wind_speed = weatherData.windSpeed[i]

        sc = solar(my_lat, my_lon, my_datestamp, my_tz, my_fmt)
        sc.compute_all()
        solar_global = zhang_huang_solar(90-sc.zenith, cloud_cover, relative_humidity,
                                         dry_bulb_present, dry_bulb_t3_hrs, wind_speed)
        Ih.append(solar_global)
        solar_altitude.append(90-sc.zenith)

    weatherData['Ih'] = Ih
    weatherData['solar_altitude'] = solar_altitude

    # split global solar radiation (GHI) into DNI and DHI
    solar_altitude = weatherData['solar_altitude'].tolist()
    doys = weatherData['doys'].tolist()
    cloud_cover = (weatherData['cloudCover']*10).tolist()
    relative_humidity = weatherData['humidity'].tolist()
    dry_bulb_present = weatherData['temperature'].tolist()
    dry_bulb_t3_hrs = weatherData['temperature_3'].tolist()
    wind_speed = weatherData['windSpeed'].tolist()
    atm_pressure = (weatherData['pressure']*100).tolist()

    # call zhang-huang solar split model
    dir_norm_rad, dif_horiz_rad = zhang_huang_solar_split(solar_altitude, doys, cloud_cover, relative_humidity,
                                                          dry_bulb_present, dry_bulb_t3_hrs, wind_speed,
                                                          atm_pressure, use_disc=False)
    weatherData['DNI'] = dir_norm_rad
    weatherData['DHI'] = dif_horiz_rad
    weatherData.to_csv('data/300SW/300SW_weather_solar.csv')
