import pandas as pd
from geopy.geocoders import Nominatim
import numpy as np
import geoplot as gplt
import geopandas as gpd
import matplotlib.pyplot as plt
import requests
import json

from scipy.spatial import cKDTree
from shapely import LineString
from shapely.geometry import Point
from shapely.wkt import loads


clean_city_data = True
clean_airport_data = True
generate_car_travel_matrix = True
generate_airport_travel_data = True
get_airline_route_data = True
visualize_data = True
visualize_airline_routes = True
initialize_slots = True

city_data_path = r'data\nhgis0007_csv\nhgis0007_ds249_20205_place.csv'
cleaned_city_data_path = r'data\city_data.csv'
airport_data_path = r'data\airport\us-airports.csv'
cleaned_airport_data_path = r'data\airport_data.csv'

car_travel_matrix_path = r'data\car_travel_matrix.csv'

minimum_population = 500000
minimum_score = 5410 #650


def read_csv(path):
    return pd.read_csv(path, encoding='latin-1')

def write_to_csv(df, path):
    return df.to_csv(path, index=False)


### City data cleaning

def remove_small_cities(df, min_pop):
    return df[df['AMPKE001'] > min_pop]

def remove_redundant_data(df):
    return df.dropna(axis=1).drop(['GISJOIN', 'YEAR', 'STUSAB', 'GEOID'], axis=1)


def group_age_brackets(df):
    df['Population 15-24'] = df.loc[:, 'AMPKE006':'AMPKE010'].sum(axis=1) + df.loc[:, 'AMPKE030':'AMPKE034'].sum(axis=1)
    df['Population 25-44'] = df.loc[:, 'AMPKE011':'AMPKE014'].sum(axis=1) + df.loc[:, 'AMPKE035':'AMPKE038'].sum(axis=1)
    df['Population 45-64'] = df.loc[:, 'AMPKE015':'AMPKE019'].sum(axis=1) + df.loc[:, 'AMPKE039':'AMPKE043'].sum(axis=1)
    df['Population 65+'] = df.loc[:, 'AMPKE020':'AMPKE025'].sum(axis=1) + df.loc[:, 'AMPKE044':'AMPKE049'].sum(axis=1)
    return df


def rename_columns(df):
    df = df.rename(columns={"AMPVE001": "Total population", "AMR8E001": "Median income"})
    df = df.drop(df.loc[:, 'AMPKE001':'AMPKE049'].columns, axis=1)
    df = df.drop(df.loc[:, 'NAME_M':'AMR8M001'].columns, axis=1)
    return df

def get_coordinates(df):
    geolocator = Nominatim(user_agent="assignment", timeout=5)
    count = 0
    for index, row in df.iterrows():
        count = count + 1
        print(count, len(df))
        loc = geolocator.geocode(str(df.loc[index, 'PLACE']) + ', ' + str(df.loc[index, 'STATE'] + ', United States'))
        if loc is None:
            df.at[index, 'Latitude'] = np.nan
            df.at[index, 'Longitude'] = np.nan
        else:
            df.at[index, 'Latitude'] = loc.latitude
            df.at[index, 'Longitude'] = loc.longitude
    df = df.dropna(axis=0)
    return df

if clean_city_data:
    data = read_csv(city_data_path)
    data = remove_small_cities(data, minimum_population)
    data = remove_redundant_data(data)
    data = group_age_brackets(data)
    data = rename_columns(data)
    data = get_coordinates(data)
    write_to_csv(data, cleaned_city_data_path)
    print('City data cleaned')

    ### City data cleaning


def remove_smallest_airports(df, min_score):
    df = df.dropna(axis=1).drop(0).dropna(axis=0)
    return df[df['score'].astype(int) > min_score]


if clean_airport_data:
    data = read_csv(airport_data_path)
    iata_data = read_csv(r'data\airport\airports.csv')
    iata_data['iata'].dropna(axis=0)
    data = remove_smallest_airports(data, minimum_score)
    for i, r in data.iterrows():
        data.loc[i, 'iata'] = iata_data[iata_data['icao'] == data.loc[i, 'ident']]['iata'].values[0]
    print(data['iata'])
    data.to_csv(cleaned_airport_data_path)
    print('Airport data cleaned')

def get_airline_routes():
    routes_path = r'data\routes.dat'
    df = pd.read_csv(routes_path, header=None, engine='python')
    df.columns = ['IATA_AIRLINE', 'ID_AIRLINE', 'IATA_DEP_AIRPORT', 'ID_DEP_AIRPORT', 'IATA_ARR_AIRPORT', 'ID_ARR_AIRPORT',
                  'CODESHARE', 'STOPS', 'AIRCRAFT_TYPE']
    return df

if get_airline_route_data:
    airline_route = get_airline_routes()
    airport_geocoded = pd.read_csv(r'data\airports_geocoded.csv')
    airport_geocoded['geometry'] = airport_geocoded['geometry'].apply(loads)
    airline_route = airline_route[airline_route['IATA_DEP_AIRPORT'].isin(set(airport_geocoded['iata']))]
    airline_route = airline_route[airline_route['IATA_ARR_AIRPORT'].isin(set(airport_geocoded['iata']))]
    dep_airport = gpd.GeoDataFrame({'iata': np.zeros(len(airline_route)), 'geometry': np.zeros(len(airline_route))})
    arr_airport = gpd.GeoDataFrame({'iata': np.zeros(len(airline_route)), 'geometry': np.zeros(len(airline_route))})
    count = 0
    for i, row in airline_route.iterrows():
        airline_route.loc[i, 'dep_geometry'] = \
        airport_geocoded[airport_geocoded['iata'] == airline_route.loc[i, 'IATA_DEP_AIRPORT']]['geometry'].values[0]
        airline_route.loc[i, 'arr_geometry'] = \
        airport_geocoded[airport_geocoded['iata'] == airline_route.loc[i, 'IATA_ARR_AIRPORT']]['geometry'].values[0]

        dep_airport.loc[count, 'iata'] = airline_route.loc[i, 'IATA_DEP_AIRPORT']

        # Instead of creating a new GeoSeries, directly assign the geometry object
        dep_airport.loc[count, 'geometry'] = airline_route.loc[i, 'dep_geometry']
        # Same for arr_airport
        arr_airport.loc[count, 'geometry'] = airline_route.loc[i, 'arr_geometry']
        count = count + 1

    dep_airport = dep_airport.set_geometry('geometry')
    dep_airport.crs = 'EPSG:4326'
    arr_airport = arr_airport.set_geometry('geometry')
    arr_airport.crs = 'EPSG:4326'
    dist = (dep_airport.to_crs(crs=3857)).distance(arr_airport.to_crs(crs=3857))
    airline_route['distance'] = dist.values
    write_to_csv(airline_route, 'data/airline_routes.csv')

    # aircraft = pd.read_csv('data/aircraft/aircraft.csv', index_col=0)
    # for index, row in airline_route.iterrows():
    #     print(aircraft.loc[aircraft['Range'] > airline_route['distance'][index]])
    #     airline_route.loc[index,'AIRCRAFT_TYPE'] = (aircraft.loc[aircraft['Range'] > airline_route['distance'][index]]).sample(1)['Type']
    #     airline_route.iloc[index, 'AIRCRAFT_TYPE']
    # print("Received airline route data")


def get_routing_data(lat1, lon1, lat2, lon2):
    r = requests.get(f"http://router.project-osrm.org/route/v1/car/{lon1},{lat1};{lon2},{lat2}"
                     f"?overview=simplified&geometries=geojson")
    return json.loads(r.content)


def calculate_travel_duration(lat1, lon1, lat2, lon2):
    routes = get_routing_data(lat1, lon1, lat2, lon2)
    return routes.get("routes")[0]['duration'] / 3600


def define_coordinates(lat1, lon1, lat2, lon2):
    routes = get_routing_data(lat1, lon1, lat2, lon2)
    return routes.get("routes")[0]['geometry']['coordinates']


def ckdnearest(gdA, gdB):
    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=True)
    gdf = pd.concat(
        [
            gdA.reset_index(drop=True),
            gdB_nearest,
            pd.Series(dist, name='dist')
        ],
        axis=1)
    return gdf

if generate_car_travel_matrix or generate_airport_travel_data or visualize_data:
    data = read_csv(cleaned_city_data_path)

    fig, ax = plt.subplots(figsize=(18, 10))

    contiguous_usa = gpd.read_file(gplt.datasets.get_path('contiguous_usa'))
    contiguous_usa.boundary.plot(ax=ax, color='gray', linewidth=0.8, zorder=-1)

    cities_points = data.apply(
        lambda srs: Point(float(srs['Longitude']), float(srs['Latitude'])),
        axis='columns'
    )
    cities_geocoded = gpd.GeoDataFrame(data, geometry=cities_points)
    cities_geocoded.crs = "EPSG:4326"
    cities_geocoded.plot(ax=ax, color='blue', markersize=30, zorder=1)

    # Create an empty travel matrix
    travel_matrix = pd.DataFrame(0, index=range(len(data)), columns=data.index)
    # Iterate over the indices and calculate travel duration
    for i in range(len(data)):
        print(i)
        lat1, lon1 = data.loc[i, 'Latitude'], data.loc[i, 'Longitude']
        latitudes = data.loc[i + 1:, 'Latitude']
        longitudes = data.loc[i + 1:, 'Longitude']
        if len(latitudes) > 0 and len(longitudes) > 0:
            if generate_car_travel_matrix:
                durations = np.vectorize(calculate_travel_duration)(lat1, lon1, latitudes, longitudes)
                travel_matrix.iloc[i + 1:, i] = durations
            elif visualize_data:
                for j in range(len(data)-i-1):
                    coordinates = define_coordinates(lat1, lon1, latitudes.iloc[j], longitudes.iloc[j])
                    line = LineString(coordinates)
                    line_gdf = gpd.GeoDataFrame(geometry=[line])
                    line_gdf.crs = "EPSG:4326"
                    line_gdf.plot(ax=ax, color='red', linewidth=1.8, zorder=0)

    # Set the diagonal values to zero
    travel_matrix = travel_matrix + travel_matrix.T - np.diag(np.diag(travel_matrix))
    for n in range(len(data)):
        travel_matrix.rename(columns={n: cities_geocoded['PLACE'][n]}, inplace=True)
        travel_matrix.set_index(cities_geocoded['PLACE'], inplace=True)
    travel_matrix.to_csv(car_travel_matrix_path)
    print(travel_matrix)

    if generate_airport_travel_data or visualize_data:
        airport_data = read_csv(cleaned_airport_data_path)
        airport_data = airport_data[airport_data['scheduled_service'] > 0]
        airport_points = airport_data.apply(
            lambda srs: Point(float(srs['longitude_deg']), float(srs['latitude_deg'])),
            axis='columns'
        )
        airports_geocoded = gpd.GeoDataFrame(airport_data, geometry=airport_points)
        airports_geocoded.crs = "EPSG:4326"
        airports_geocoded.plot(ax=ax, color='green', markersize=30, zorder=1)
        airports_geocoded.to_csv(r'data\airports_geocoded.csv')
        airport_to_city = ckdnearest(cities_geocoded, airports_geocoded)
        airport_to_city = airport_to_city[['iata', 'PLACE', 'longitude_deg', 'latitude_deg', 'Longitude', 'Latitude']]
        durations = []
        for n, row in airport_to_city.iterrows():
            lat1, lon1 = airport_to_city.loc[n, 'Latitude'], airport_to_city.loc[n, 'Longitude']
            lat2, lon2 = airport_to_city.loc[n, 'latitude_deg'], airport_to_city.loc[n, 'longitude_deg']
            coordinates = define_coordinates(lat1, lon1, lat2, lon2)
            durations.append(np.vectorize(calculate_travel_duration)(lat1, lon1, lat2, lon2))
            line = LineString(coordinates)
            line_gdf = gpd.GeoDataFrame(geometry=[line])
            line_gdf.crs = "EPSG:4326"
            line_gdf.plot(ax=ax, color='orange', linewidth=1.8, zorder=0)
        airport_to_city['duration'] = durations
        airport_to_city.to_csv('data/airport_city_match')

    airport_distance_matrix = pd.DataFrame(0, index=range(len(airports_geocoded)), columns=airports_geocoded['iata'])
    airports_geocoded.crs = 'EPSG:4326'
    airports_geocoded = airports_geocoded.to_crs(crs=3857)

    for n in range(len(airports_geocoded)):
        airport_distance_matrix.set_index(airports_geocoded['iata'], inplace=True,)
    airport_distance_matrix.index.name = None
    i=0
    for i in range(len(airports_geocoded)):
        for n in range(len(airports_geocoded)):
            distance = airports_geocoded.iloc[i]['geometry'].distance(airports_geocoded.iloc[n]['geometry'])
            airport_distance_matrix.iloc[i, n] = distance
    airport_distance_matrix.to_csv('data/airport_distance_matrix.csv')


    airline_route = airline_route.reset_index()
    if visualize_airline_routes and get_airline_route_data:
        for n, row in airline_route.iterrows():
            line = LineString([airline_route.loc[n, 'dep_geometry'], airline_route.loc[n, 'arr_geometry']])
            line_gdf = gpd.GeoDataFrame(geometry=[line])
            line_gdf.crs = "EPSG:4326"
            line_gdf.plot(ax=ax, color='blue', linewidth=0.5, zorder=0, alpha = 0.2)

        plt.tight_layout()
        plt.savefig('plot.png', dpi=600)






