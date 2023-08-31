import random

import numpy as np
import pandas as pd
from functools import reduce


def read_csv(path):
    return pd.read_csv(path, encoding='latin-1')


city_data = pd.DataFrame(read_csv("data/city_data.csv"))

total_population = city_data['Total population'].sum()
travelling_population = total_population * 0.0001

car_data = pd.DataFrame(read_csv("data/car_travel_matrix.csv"))
car_data = car_data.set_index('PLACE')

airport_city_match = pd.DataFrame(read_csv("data/airport_city_match"))
airport_city_match = airport_city_match.drop('Unnamed: 0', axis=1)
airport_city_match.set_index('PLACE', inplace=True)

aircraft = pd.read_csv("data/aircraft/aircraft.csv", index_col=0)
airport_distance_matrix = pd.read_csv("data/airport_distance_matrix.csv", index_col=0)
routes = pd.read_csv("data/airline_routes.csv", index_col=0)

class Pop:
    def __init__(self, location, destination, age, income):
        self.location = location
        self.destination = destination
        self.age = age
        self.income = income


def random_location():
    location = random.choices(population=city_data['PLACE'].values.tolist(),
                              weights=city_data['Total population'].values.tolist())
    return location


def random_destination():
    destination = random.choices(population=city_data['PLACE'].values.tolist(),
                                 weights=city_data['Total population'].values.tolist())
    return destination


def random_age(location):
    age = random.choices(population=['15-24', '25-44', '45-64', '65+'],
                         weights=city_data.loc[city_data['PLACE'] == location,
                                 'Population 15-24':'Population 65+'].values.flatten().tolist())
    return age


def random_income(location):
    income = random.choices(
        population=city_data.loc[city_data['PLACE'] == location, 'Median income'].values.flatten().tolist(),
        weights=city_data.loc[city_data['PLACE'] == location, 'Median income'].values.flatten().tolist())
    return income

def assign_aircraft():
    for i in range(len(routes)):
        routes.loc[i, 'Aircraft'] =(aircraft[aircraft['Range'] > routes['distance'][i]]).sample(1)['Type'].values[0]
        routes.loc[i, 'Aircraft'] =(aircraft[aircraft['Range'] > routes['distance'][i]]).sample(1)['Type'].values[0]
    return routes

routes.reset_index(inplace=True)
routes['Aircraft'] = np.zeros(len(routes))
routes = assign_aircraft()
routes.set_index(['IATA_DEP_AIRPORT', 'IATA_ARR_AIRPORT'], inplace=True)
routes = routes[~routes.index.duplicated(keep='first')]
print(routes)
aircraft = aircraft.set_index('Type')
pop = pd.DataFrame(columns=['Location', 'Departure_airport', 'Destination', 'Arrival_airport', 'Age', 'Income', 'Car travel time', 'Car travel cost', 'Air travel time',
                            'Air travel cost', 'Mode', 'Aircraft'])
time_weight = 3
cost_weight = 0.05


for i in range(int(travelling_population)):
    location = random_location()[0]
    departure_airport = airport_city_match.loc[location, 'iata']
    destination = random_destination()[0]
    arrival_airport = airport_city_match.loc[destination, 'iata']
    age = random_age(location)[0]
    income = random_income(location)[0]
    car_travel_time = car_data.loc[location].loc[destination]
    car_travel_costs = car_travel_time * 10
    try:
        ac = routes.loc[(departure_airport, arrival_airport)]['Aircraft']
    except:
        ac = 'Narrowbody'
        pass

    air_travel_time = airport_distance_matrix.loc[departure_airport, arrival_airport] / 1000 / aircraft.loc[ac, 'Speed'] \
                      + 2 + airport_city_match.loc[location, 'duration'] + airport_city_match.loc[destination, 'duration']
    air_travel_costs = aircraft.loc[ac, 'TCO/hr'] * airport_distance_matrix.loc[departure_airport, arrival_airport] \
                       / 1000 / aircraft.loc[ac, 'Speed'] * 1.2 / aircraft.loc[ac, 'PAX'] + 10
    if air_travel_costs == 10:
        air_travel_costs = 1000

    if car_travel_time * time_weight + car_travel_costs * cost_weight < air_travel_time * time_weight + air_travel_costs * cost_weight:
        mode = 'Car'
    else:
        mode = 'Air'
    pop.loc[len(pop.index)] = [location, departure_airport, destination, arrival_airport, age, income, car_travel_time, car_travel_costs,
                               air_travel_time, air_travel_costs, mode, ac]

pop.to_csv('data/pop.csv')
print(pop)


