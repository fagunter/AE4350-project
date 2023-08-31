import random
import numpy as np
import pandas as pd
import geopandas as gpd

pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
import geoplot as gplt
from shapely import LineString


def read_csv(path):
    return pd.read_csv(path, encoding='latin-1')


def write_to_csv(df, path):
    return df.to_csv(path, index=False)


def flight_pop_match(pop, state, state_id):
    for i, row in state.iterrows():
        for n, row in pop.iterrows():
            if state['IATA_ARR_AIRPORT'][i] == pop['Arrival_airport'][n] and state['IATA_DEP_AIRPORT'][i] == \
                    pop['Departure_airport'][n] and pop['Mode'][n] == 'Air':
                pop.loc[n, 'flight ID'] = str(i)
    pop['count'] = pop.groupby('flight ID')['flight ID'].transform('count')
    pop = pop.sort_values('count', ascending=False)
    popn = pop.loc[~(pop == 0).any(axis=1)]
    state['Passengers'] = 0
    idx = popn.loc[:, ['flight ID', 'count', 'Aircraft']].drop_duplicates().reset_index()
    for r in range(len(idx)):
        state['Passengers'][int(float(idx.iloc[r, 1]))] = idx.iloc[r, 2]
        if state_id == 0:
            state['Aircraft'][int(float(idx.iloc[r, 1]))] = idx.iloc[r, 3]
    state.sort_values('Passengers', ascending=False, inplace=True)
    return pop, state


airports_geocoded = read_csv('data/airports_geocoded.csv')

aircraft = read_csv('data/aircraft/aircraft.csv')

airport_city_match = read_csv('data/airport_city_match')
airport_city_match = airport_city_match.drop('Unnamed: 0', axis=1)

pop = read_csv('data/pop.csv')
pop = pop.drop('Unnamed: 0', axis=1)

state_zero = read_csv('data/airline_routes.csv').sample(15)
state_zero['Passengers'] = np.zeros(len(state_zero))
state_zero['Aircraft'] = 'Narrowbody'
pop['flight ID'] = np.zeros(len(pop))

alpha = 0.3
gamma = 0.6
epsilon = 0.3

all_epochs = []
all_penalties = []

actions = np.arange(0, 7 * len(state_zero) + 1, 1)

pop, state = flight_pop_match(pop, state_zero, 0)
q_table = np.zeros([len(state) * len(airport_city_match) ** 2 * len(aircraft), len(actions)])

i = 0
plt.figure(0)
fig, axs = plt.subplots(2, 2)

for n in range(3):
    state = state.loc[:, ['IATA_DEP_AIRPORT', 'IATA_ARR_AIRPORT', 'Passengers', 'Aircraft', 'distance']]
    state = state_zero
    next_state = state_zero
    state_id = 0
    done = False
    q_value_lst = []
    passenger_reward_lst = []
    sustainability_reward_lst = []
    reward_lst = []

    while done == False:
        if np.random.uniform(0, 1) < epsilon:
            action = random.choices(actions)[0]
        else:
            action = np.argmax(q_table[state_id])

        old_value = q_table[state_id, action]

        departure_airports = random.choices(population=airports_geocoded['iata'].values, k=20)
        arrival_airports = random.choices(population=airports_geocoded['iata'].values, k=20)
        departure_geometry = []
        arrival_geometry = []

        for idx in range(len(departure_airports)):
            departure_geometry.append(airports_geocoded.set_index('iata').loc[departure_airports[idx], 'geometry'])
            arrival_geometry.append(airports_geocoded.set_index('iata').loc[arrival_airports[idx], 'geometry'])

        departure_airports = pd.DataFrame(departure_airports, columns=['DEP'])
        arrival_airports = pd.DataFrame(arrival_airports, columns=['ARR'])
        airports = pd.concat([departure_airports, arrival_airports], axis=1)
        departure_airports['geometry'] = gpd.GeoSeries.from_wkt(departure_geometry)
        departure_airports = gpd.GeoDataFrame(departure_airports, geometry='geometry')
        arrival_airports['geometry'] = gpd.GeoSeries.from_wkt(arrival_geometry)
        arrival_airports = gpd.GeoDataFrame(arrival_airports, geometry='geometry')
        departure_airports.crs = 'EPSG:4326'
        arrival_airports.crs = 'EPSG:4326'
        airports['DIST'] = (departure_airports.to_crs(crs=3857)).distance(arrival_airports.to_crs(crs=3857))
        count = 0
        length = len(state)
        if 0 <= action < length:
            airports = (airports.sort_values('DIST', ascending=True).loc[~(airports == 0).any(axis=1)]).reset_index()
            airports = airports.loc[~(airports == 0).all(axis=1)]
            state.reset_index(inplace=True)
            state.loc[action, 'IATA_ARR_AIRPORT'] = airports['ARR'][0]
            state.loc[action, 'IATA_DEP_AIRPORT'] = airports['DEP'][0]
            state.loc[action, 'distance'] = airports['DIST'][0]
            state.set_index('index', drop=True, inplace=True)
            pop, state = flight_pop_match(pop, state, state_id)


        elif length <= action < 2 * length:
            airports = (airports.sort_values('DIST', ascending=False).loc[~(airports == 0).any(axis=1)]).reset_index()
            airports = airports.loc[~(airports == 0).all(axis=1)]
            state.reset_index(inplace=True)
            state.loc[action - length, 'IATA_ARR_AIRPORT'] = airports['ARR'][len(airports) - 1]
            state.loc[action - length, 'IATA_DEP_AIRPORT'] = airports['DEP'][len(airports) - 1]
            state.loc[action - length, 'distance'] = airports['DIST'][len(airports) - 1]
            state.set_index('index', drop=True, inplace=True)
            pop, state = flight_pop_match(pop, state, state_id)

        elif 2 * length <= action < 3 * length:
            state.reset_index(inplace=True)
            state.loc[action - 2 * length, 'Aircraft'] = aircraft.iloc[0, 1]
            state.set_index('index', drop=True, inplace=True)

        elif 3 * length <= action < 4 * length:
            state.reset_index(inplace=True)
            state.loc[action - 3 * length, 'Aircraft'] = aircraft.iloc[1, 1]
            state.set_index('index', drop=True, inplace=True)

        elif 4 * length <= action < 5 * length:
            state.reset_index(inplace=True)
            state.loc[action - 4 * length, 'Aircraft'] = aircraft.iloc[2, 1]
            state.set_index('index', drop=True, inplace=True)

        elif 5 * length <= action < 6 * length:
            state.reset_index(inplace=True)
            state.loc[action - 5 * length, 'Aircraft'] = aircraft.iloc[3, 1]
            state.set_index('index', drop=True, inplace=True)

        elif 6 * length <= action < 7 * length:
            state.reset_index(inplace=True)
            state.loc[action - 6 * length, 'Aircraft'] = aircraft.iloc[4, 1]
            state.set_index('index', drop=True, inplace=True)

        next_state = state
        state_id = state_id + 1

        passenger_reward = next_state['Passengers'].sum() / len(pop) * 100

        sustainability_reward = (len(next_state[next_state['Aircraft'].str.contains('Hydrogen-electric regional')]) + \
                                 len(next_state[next_state['Aircraft'].str.contains('Battery-electric regional')]) + \
                                 len(next_state[next_state['Aircraft'].str.contains('Narrowbody')])) / 15
        feasibility_reward = 0
        next_state.reset_index(inplace=True)
        for i in range(len(next_state)):
            if aircraft.loc[
                aircraft.index[aircraft['Type'] == next_state.loc[i, 'Aircraft']], 'Range'].values.size > 0 and \
                    next_state.loc[i, 'distance'] > aircraft.loc[
                aircraft.index[aircraft['Type'] == next_state.loc[i, 'Aircraft']], 'Range'].values:
                feasibility_reward = feasibility_reward - 1
        next_state.set_index('index', drop=True, inplace=True)
        reward = passenger_reward + sustainability_reward + feasibility_reward
        next_max = np.max(q_table[state_id])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state_id, action] = new_value

        print(reward, new_value, passenger_reward, sustainability_reward)
        if next_state['Passengers'].sum() > 0.1 * len(pop) or state_id > 500:
            done = True

        q_value_lst.append(new_value)
        passenger_reward_lst.append(passenger_reward * len(pop) / 100)
        sustainability_reward_lst.append(sustainability_reward * 15)
        reward_lst.append(reward)

        # if reward == -10:
        #     penalties += 1

        # epochs += 1
    print(n)
    q_value_lst = pd.DataFrame(q_value_lst)
    reward_lst = pd.DataFrame(reward_lst)
    passenger_reward_lst = pd.DataFrame(passenger_reward_lst)
    sustainability_reward_lst = pd.DataFrame(sustainability_reward_lst)

    next_state.to_csv(f'data/states/state{n}')

    q_value_lst.to_csv(f'data/states/q_val{n}')
    reward_lst.to_csv(f'data/states/reward{n}')
    passenger_reward_lst.to_csv(f'data/states/pas_r{n}')
    sustainability_reward_lst.to_csv(f'data/states/sus_r{n}')

    axs[0, 0].plot(q_value_lst)
    axs[0, 0].set_title('Q-value')
    axs[0, 1].plot(reward_lst)
    axs[0, 1].set_title('Reward')
    axs[1, 0].plot(passenger_reward_lst)
    axs[1, 0].set_title('Passengers')
    axs[1, 1].plot(sustainability_reward_lst)
    axs[1, 1].set_title('Sustainable Aircraft')

    for ax in axs.flat:
        ax.set(xlabel='Step', ylabel='Variable')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    fig.tight_layout(pad=1.0)
plt.show()

# Hide x labels and tick labels for top plots and y ticks for right plots.
state.reset_index(inplace=True)
# plt.figure(1)

# contiguous_usa = gpd.read_file(gplt.datasets.get_path('contiguous_usa'))
# contiguous_usa.boundary.plot(ax=ax, color='gray', linewidth=0.8, zorder=-1)

# for i in range(len(state)):
#     print([departure_airports.loc[i, 'geometry'], arrival_airports.loc[i,'geometry']])
#     line = LineString([departure_airports.loc[i, 'geometry'], arrival_airports.loc[i,'geometry']])
#     line_gdf = gpd.GeoDataFrame(geometry=[line])
#     line_gdf.crs = "EPSG:4326"
#     line_gdf.plot(ax=ax, color='red', linewidth=1.8, zorder=0)
# state.set_index('index', inplace=True, drop=True)
# plt.show()
#
# print(q_table)
