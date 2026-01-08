#Author: Energy and Technology Policy Group, ETH last edited by Churchill Agutu
# Python version 2021 (3.8.8)
#Acknowledgements - ChatGPT was used as an aid to generate the code descriptions only
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import scipy
import sys
from scipy.stats.stats import pearsonr
import seaborn as sns

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


"""
Main code backbone
"""

def obtain_load_profile(data, GHI_data):
    """
    This function takes in a dataframe with individual loads and GHI data, and returns a list of dataframes that includes
    the individual load profiles, an aggregated load profile, the number of loads, and the pearson coefficient between
    the aggregated load and the GHI data.

    :param data: A dataframe that includes individual load profiles.
    :param GHI_data: A dataframe that includes GHI data.
    :return: A list of dataframes that includes concatenated individual load profiles, an aggregated load profile, the
             number of loads, and the pearson coefficient between the aggregated load and the GHI data.
    """

    #read load profile from defined directory
    #data is sourced as individual load profiles
    #Create empty dataframe

    sum_of_loads = pd.DataFrame(columns =['load'])

    #load_profile columns
    load_profile_columns = data.drop(['Unnamed: 0'],axis = 1) #convert to float #need to find way to pick only specific columns #convert to kW

    # timestamps_column = load_profile['time']
    sum_of_loads["load"] = load_profile_columns.sum(axis =1)
    aggregate_load_profile = pd.concat([sum_of_loads], axis = 1)

    print (sum_of_loads['load'].sum())

    #create df for individual loads
    concatenated_ind_load_profiles= []

    for i in range(load_profile_columns.shape[1]): #no.of columns
        # concatenated_ind_load_profiles_new = concatenated_ind_load_profiles
        #create df of individual load profiles
        individual_load_profile = pd.DataFrame([load_profile_columns.iloc[:,i]]).transpose() #Convert to kW
        concatenated_ind_load_profiles.append(individual_load_profile)

    # pearson coefficient calculation
    demand_values = aggregate_load_profile['load'].head(72)  # use 3 days of data for correlations
    GHI = GHI_data['GHI'].head(72)  # obtain corresponding rows for GHI
    updated_df = pd.concat([demand_values, GHI], axis=1)
    pearsonr_coeff, p_value = pearsonr(GHI, demand_values)  # calculate coefficient

    #return number of loads
    num_of_loads = len(load_profile_columns.columns)
    print ("Number of loads per site",num_of_loads)

    # aggregate_load_profile.head(24).plot()
    # plt.show()


    return [concatenated_ind_load_profiles, aggregate_load_profile,num_of_loads,pearsonr_coeff] #list of dataframes


def pv_load (module_size, num_modules, Tamb, GHI, NOCT, MPPT):

    """
    The function calculates available power from PV each hour

    :param module_size: Size of module in #kW
    :param num_modules: unit
    :param Tamb: Ambient temperature at hour #h
    :param GHI_dataframe: Global irradiation at hour h in #W/m^2
    :param NOCT: Nominal Operating Cell Temperature #degC
    :return: total PV capacity, hourly PV load #kW
    """
    cell_temp = Tamb + (GHI/800) * (NOCT - 20) #adjust cell temperature to new GHI conditions # 800 is the GHI  at standard conditions (henry louise pg 197)
    temperature_factor = 1 + MPPT *(cell_temp - 25) #Osterwalds method #adjusts  MPPT output for varryng temperature
    total_pv_capacity = module_size * num_modules
    pv_load_per_hour = (total_pv_capacity * temperature_factor * GHI/1000)

    #
    return [total_pv_capacity,pv_load_per_hour]

def battery_specs (tech_type, charge_efficiency, discharge_efficiency, battery_capacity,battery_power, battery_max_soc, battery_min_soc, annual_batt_cost_reduction, battery_life):
    """
    The function is used to define the specifications of a battery for use in a system.
    It takes in various parameters such as the battery technology type,
    charge and discharge efficiency, battery capacity, power,
    maximum and minimum state of charge (SOC),
    annual battery cost reduction, and battery life in years.

    :param tech_type: lead acid, lithium ion
    :param charg_efficiency:
    :param discharge_efficiency:
    :param battery_capacity: kWh
    :param battery_power: kW
    :param battery_max_SOC: %
    :param battery_min_SOC: %
    :param annual_batt_cost_reduction: %
    :param battery_life: years
    :return: tech_type, charge_efficiency, discharge_efficiency, battery_capacity,battery_power, batt_cap_max_soc, batt_cap_min_soc, annual_batt_cost_reduction,battery_life
    """
    # battery
    battery_DoD_buffer = 0.9  # account for limit in battery Depth of Discharge at battery end of life
    battery_SOH = 0.8  # battery SOH (State of Health) at end of life
    actual_battery_capacity = battery_capacity * battery_SOH * battery_DoD_buffer
    batt_cap_max_soc = battery_max_soc * actual_battery_capacity
    batt_cap_min_soc = battery_min_soc * actual_battery_capacity

    return [tech_type, charge_efficiency, discharge_efficiency, battery_capacity,battery_power, batt_cap_max_soc, batt_cap_min_soc, annual_batt_cost_reduction,battery_life]


def inverter_specs (inverter_eff, inverter_cost, annual_inv_cost_reduction, inverter_life):
    """
    Define the specifications of an inverter for use in a system.

    :param inverter_eff: %
    :param inverter_cost: USD/kW
    :param annual_inv_cost_reduction: %
    :param inverter_life: years
    :return: inverter_eff, inverter_cost, inverter_life, annual_inv_cost_reduction
    """
    return[inverter_eff, inverter_cost, inverter_life, annual_inv_cost_reduction]


def gen_capacity_sizing(load_profile, pv_load, inverter_eff, tech_type, charge_eff, discharge_eff, batt_capacity, batt_cap_max_soc, batt_cap_min_soc):
    """
    This function performs capacity sizing for a battery system based on the load profile,
    hourly PV capacities, inverter efficiency, technology type, charge efficiency,
    discharge efficiency, battery capacity, maximum state of charge, and minimum
    state of charge. The function calculates the residual load and carries out an
    energy balance at each time step. It uses an algorithm to dispatch excess PV power
    or excess load to the battery system.

    The function defines initial conditions and limits and uses if statements to check
    for different scenarios (excess PV, excess load, no load). It then calculates the
    battery capacity and unmet load for each time step, and appends the results to
    various lists. Finally, the function returns the loss of energy probability
    and total energy per year.

    :param load_profile: dataframe with hourly loads for a single load # kW
    :param pv_load: array with list of hourly PV capacities # kW
    :param inverter_eff: inverter efficiency #%
    :param tech_type:  user defined e.g. SAS, MG
    :param charge_eff: user defined #%
    :param discharge_eff: user defined #%
    :param batt_capacity: user defined #kWh
    :param batt_cap_max_soc: user defined #kWh
    :param batt_cap_min_soc: user defined #kWh
    :return: loss_of_energy_probability,total_energy_per_year
    """
    # source inputs
    # pv_load  is already an array no need to convert
    # print (load_profile)
    load_profile = (load_profile.iloc[:,-1]).to_numpy() #find load column and convert to array

    #carry out energy balance at each time [t]
    timestep  = 1 #hour
    #define initial conditions
    batt_soc_cap_in = batt_cap_max_soc/2 #kWh
    full_cycle_capacity = batt_cap_max_soc *2 #charge and discharge
    batt_soc_cap_out = batt_soc_cap_in #kWhn #initial
    power_outage_count = 0
    cycle_counter = 0 # battery cycle counter
    batt_soc_cap_out_list =[]
    batt_cap_list = [] #battery capacity list
    batt_cap_list_new = []
    total_unmet_load = 0
    total_unmet_load_list = []
    residual_load_list = [] # hourly
    load_met_by_pv_list =[] #hourly
    load_met_by_battery_list =[]

    #define limits
    batt_cap_limit = 1000  #not used in model, but in case limit needs to be set in future
    # print ("bat cap LIMIT",batt_cap_limit)

    # calculate residual load # meet load using PV #last column load profile
    residual_load = load_profile  - pv_load * inverter_eff  #kW
    residual_energy= residual_load * timestep #kWh
    total_energy_per_year = (load_profile * timestep).sum() #total energy

    #Dispatch algorithm
    for i,residual_energy_t in enumerate (residual_energy):
        #update new batt_soc_out value
        batt_soc_cap_in = round(batt_soc_cap_out,6) #reasonable rounding to 6 dc pl because python struggles to sum floats with very number of decimal places

        # Excess PV
        if residual_energy_t < 0 and batt_soc_cap_in > batt_cap_max_soc:  # seventh condition #excess pv and full battery capacity or too much
            print (batt_soc_cap_in, batt_cap_max_soc)

            sys.exit("error in energy balance, battery has too much capacity")

        if residual_energy_t < 0 and batt_soc_cap_in < batt_cap_min_soc:  # seventh condition #excess pv and battery goes below min soc

            sys.exit("error in energy balance, battery has surpassed DOD limit")

        #Excess load if statements
        if residual_energy_t == 0: # first condition # no load
            batt_soc_cap_out = batt_soc_cap_in
            met_load = 0
            unmet_load = 0
            total_unmet_load_list.append(unmet_load)

        if residual_energy_t > 0 and  batt_soc_cap_in > batt_cap_min_soc and batt_soc_cap_in <= batt_cap_max_soc: # second condition - excess load and some battery capacity available
            available_battery_discharging_capacity = (batt_soc_cap_in - (batt_cap_min_soc + 0.00001)) #dispatcheable # battery side #0.0001 is because the sum doesn not calculate exact numbers all the time#so this rounds up #python artefact
            required_battery_capacity = residual_energy_t / discharge_eff / inverter_eff   #if discharging to meet load # because you need to find out how much battery capacity is needed
            dispatcheable_batt_cap_limit = batt_cap_limit# dispatcheable #battery side  #need to make them comparable

            met_load = min([dispatcheable_batt_cap_limit,required_battery_capacity,available_battery_discharging_capacity])
            batt_soc_cap_out = batt_soc_cap_in - met_load #discharge values should be higher so that you get the right amount after losses
            unmet_load = residual_energy_t - (met_load * discharge_eff * inverter_eff)  #unmet load #need to go to residual load side
            #power outage hours count
            power_outage_count = power_outage_count + 1
            #calculate unmet load
            total_unmet_load = total_unmet_load + unmet_load
            total_unmet_load_list.append(unmet_load)

            if unmet_load < - 0.000000001: #load is only unmet if residual_energy_t > 0 and is more than the available capacity #code struggles to calculate when 0
                print("unmet load cannot be negative")
                sys.exit("Error message")

        if residual_energy_t > 0 and batt_soc_cap_in == batt_cap_min_soc: #third condition - excess load & no battery capacity available
            batt_soc_cap_out = batt_soc_cap_in
            met_load = 0
            unmet_load = residual_energy_t - met_load
            power_outage_count= power_outage_count + 1
            total_unmet_load = total_unmet_load + unmet_load
            total_unmet_load_list.append(unmet_load)

        #Excess PV
        if residual_energy_t < 0 and batt_soc_cap_in == batt_cap_min_soc: #fourth condition - excess PV and no battery capacity available
            possible_battery_charging_capacity = min((batt_cap_max_soc - batt_cap_min_soc), batt_cap_limit) # in case the charge capacity (see below) is more than the limit
            charge_capacity = abs(residual_energy_t)* charge_eff * inverter_eff #charge battery from PV #from residual side
            batt_soc_cap_out = batt_cap_min_soc + min(np.array([charge_capacity, possible_battery_charging_capacity])) #charge battery with excess PV capacity
            unmet_load = 0
            total_unmet_load_list.append(unmet_load)

        if residual_energy_t < 0 and batt_soc_cap_in == batt_cap_max_soc: #fifth condition #excess pv and full battery capacity or too much
            dump_load = residual_energy_t #dump excess load
            batt_soc_cap_out = batt_soc_cap_in
            unmet_load = 0
            total_unmet_load_list.append(unmet_load)

        if residual_energy_t < 0 and batt_soc_cap_in > batt_cap_min_soc and batt_soc_cap_in < batt_cap_max_soc: #sixth condition - excess pv and battery not fully charged
            total_required_battery_charging_capacity = batt_cap_max_soc - batt_soc_cap_in #amount of energy to charge battery to full capacity
            available_charging_from_pv = abs(residual_energy_t)* charge_eff * inverter_eff #amount of capacity available from PV
            required_battery_charging_capacity_from_pv = total_required_battery_charging_capacity
            # print ("added capacity", min(batt_cap_limit,abs(residual_energy_t),required_battery_charging_capacity) * charge_eff * inverter_eff )
            batt_soc_cap_out = batt_soc_cap_in + min(batt_cap_limit,available_charging_from_pv,required_battery_charging_capacity_from_pv)  #charge battery
            unmet_load = 0
            total_unmet_load_list.append(unmet_load)

        #Append battery SOC data to list
        batt_soc_cap_out_list.append(batt_soc_cap_out)

    try:
        loss_of_energy_probability  =  (total_unmet_load/total_energy_per_year)

    except ZeroDivisionError: # in case there are columns with no loads
        loss_of_energy_probability = 99 # 99 just shows that it did not meet load

    return [loss_of_energy_probability,total_energy_per_year]
#
def distribution_sizing(distance_between_loads, distribution_losses, num_of_loads):

    """
    The function is used to generate properties of the distribution infrastructure for mini-grids

    :param distance_between_loads: user defined #km
    :param distribution_losses: user defined #%
    :param num_of_loads: calculated using the obtain_load_profile function
    :return: lv_line_length, mv_line_length, lv_line_life, mv_line_life,distribution_losses,num_of_loads,transformer_life
    """
    lv_line_length = distance_between_loads * num_of_loads
    mv_line_length = 0
    lv_line_life = 20 #years
    mv_line_life = 20 #years
    transformer_life = 25 #years
    return [lv_line_length, mv_line_length, lv_line_life, mv_line_life,distribution_losses,num_of_loads,transformer_life]

#reduced cost function
def reduced_tech_cost (start_year, reinvestment_year, tech_cost_per_unit_start_year,learning_rate): # assuming negative exponential function
    """
    The function is used to estimate the cost of a technology component when it is replaced at the end of its life
    :param start_year: initial year when the cost of the system was determined
    :param reinvestment_year: year to - reinvest
    :param tech_cost_per_unit_start_year: intial cost
    :param learning_rate: %
    :return: cost at the reinvestment year
    """
    year = np.arange(start_year - start_year,(reinvestment_year + 1) - start_year, 1)
    year_o_tech_cost = tech_cost_per_unit_start_year
    tech_cost_year_n = year_o_tech_cost * (1 - learning_rate)**year

    return tech_cost_year_n [reinvestment_year - start_year] #system cost at year of re-investment

#calculate LCOE
def lcoe_calculation(start_year,tech_architecture,total_energy_per_year, project_life, pv_life, battery_life, inverter_life, num_connections, cost_of_capital,
                     lv_line_length, lv_line_life,mv_line_length,mv_line_life,distribution_losses, pv_capacity, battery_capacity, battery_power, inverter_size, num_of_transformers, transformer_life,
                     transformer_cost, pv_cost, annual_pv_cost_reduction, balance_of_system_cost_factor, battery_cost,annual_batt_cost_reduction,inverter_cost, annual_inverter_cost_reduction,
                     o_m_costs_generation_system, o_m_costs_distribution_system, lv_line_cost, mv_line_cost,cost_per_connection):

    """
    The function first determines whether the project is a minigrid or a standalone
    system and calculates the investment costs for the distribution infrastructure
    accordingly. It then calculates the investment costs for each equipment in the
    project, including PV panels, batteries, inverters, and the distribution system,
    and updates arrays with these costs over the project life. Finally, it calculates
    the LCOE using these investment costs and the operational and maintenance costs
    of the generation and distribution systems.

    :param start_year: initial year when investment is made
    :param tech_architecture: system coupling  e.g AC-AC, AC-DC
    :param total_energy_per_year: total energy consumed #kWh/year
    :param project_life: time over which the system is operational
    :param pv_life: PV life time #years
    :param battery_life: battery lifetime #years
    :param inverter_life: inverter lifetime #years
    :param num_connections: number of loads #years
    :param cost_of_capital: nominal cost of captial #years
    :param lv_line_length: Low Voltage line length #km
    :param lv_line_life: Low Voltage line lifetime #years
    :param mv_line_length: Medium Voltage line length #km
    :param mv_line_life: Medium Voltage line lifetime #years
    :param distribution_losses: % technical losses of distribution line
    :param pv_capacity: PV panel total capacity #kWh
    :param battery_capacity: Battery total capacity #kWh
    :param battery_power: Battery power #kW
    :param inverter_size: Inverter size #kW
    :param num_of_transformers: No. of units #kW
    :param transformer_life: Transformer life # years
    :param transformer_cost: Transformer cost #USD/kW
    :param pv_cost: PV cost #USD/kW
    :param annual_pv_cost_reduction: %/year
    :param balance_of_system_cost:
    :param battery_cost: USD/kWh
    :param annual_batt_cost_reduction: %/year
    :param inverter_cost: #USD/kW
    :param annual_inverter_cost_reduction: %/year
    :param o_m_costs_generation_system: % of total system invesment costs
    :param o_m_costs_distribution_system: % of total distribution system invesment cost
    :param lv_line_cost: #USD/km
    :param mv_line_cost: #USD/km
    :param cost_per_connection: #USD/connection
    :return: LCOE #USD/kWh
    """
    # Determine if you need distribution infrastructure costs for calculations #ignore learning curves for distribution infrastructure since they will likely last ca. 20 years +
    if tech_architecture == "MG":
        lv_line_investment_cost = (lv_line_length /(1 - distribution_losses)) * lv_line_cost
        mv_line_investment_cost = (mv_line_length / (1 - distribution_losses)) * mv_line_cost
        transformer_investment_cost = (num_of_transformers * transformer_cost)
        total_connection_cost = (num_connections * cost_per_connection)

    else:  # standalone systems
        lv_line_investment_cost = 0
        mv_line_investment_cost = 0
        transformer_investment_cost = 0
        total_connection_cost = 0

    if total_energy_per_year > 500: #(probably SHS)

        inverter_cost = inverter_cost

    else:#no inverter
        inverter_cost = 0

    #Annual_costs
    #Annual Investment costs
    #define arrays for each equipment
    pv_investment_cost_array= np.zeros(project_life+1)
    batt_investment_cost_array = np.zeros(project_life + 1)
    inv_investment_cost_array = np.zeros(project_life + 1)
    balance_of_system_investment_cost_array = np.zeros(project_life + 1)
    dist_investment_cost_array = np.zeros(project_life + 1)
    year = np.arange(0, project_life + 1, 1)  # create data array for year

    #define investment cost in year 0
    pv_investment_cost = (pv_cost * pv_capacity)  # USD
    battery_investment_cost = (battery_capacity * battery_cost)   # USD #c
    inverter_investment_cost = (inverter_size * inverter_cost)  # USD
    # balance_of_system_cost_investment_cost = (balance_of_system_cost * pv_capacity) #USD
    distribution_investment_cost = (lv_line_investment_cost + mv_line_investment_cost + transformer_investment_cost + total_connection_cost)

    balance_of_system_cost_investment_cost = balance_of_system_cost_factor * (pv_investment_cost + battery_investment_cost + inverter_investment_cost + distribution_investment_cost) # BOS is a fraction of the system costs

    #update array with investment cost numbers
    pv_investment_cost_array [0] = pv_investment_cost #USD
    batt_investment_cost_array[0] = battery_investment_cost #USD
    inv_investment_cost_array[0] = inverter_investment_cost #USD
    dist_investment_cost_array[0] = distribution_investment_cost  # USD
    balance_of_system_investment_cost_array [0]= balance_of_system_cost_investment_cost #USD #no_learnings assumed

    # create list of tech_lives
    tech_life_dict = {'pv_life': pv_life, 'battery_life': battery_life, 'inverter_life': inverter_life, 'lv_line_life':lv_line_life, 'mv_line_life': mv_line_life, 'transformer_life': transformer_life}

    # update investment cost in the respective year based on tech life and tech cost reductions
    for equipment, tech_life in tech_life_dict.items():
        num_of_reinvesment_years = math.floor(project_life/ (tech_life+1))  # calculate how many times you have to re-invest in tech #round down - realistic number

        for i in range(num_of_reinvesment_years):  # add investment cost based on reinvesment year
            # calculate new investment cost based on technology annual cost reduction
            reinvestment_year = start_year + tech_life * (i + 1)  # python starts from zero # so exclude re-investment year 0

            if reinvestment_year <= project_life:
                if equipment == 'pv_life':
                    new_pv_cost = pv_cost
                    pv_investment_cost_array[reinvestment_year] = new_pv_cost * pv_capacity

                if equipment == 'battery_life':
                    new_batt_cost = reduced_tech_cost (start_year = start_year, reinvestment_year = reinvestment_year, tech_cost_per_unit_start_year = battery_cost,learning_rate = annual_batt_cost_reduction)
                    batt_investment_cost_array[reinvestment_year] = new_batt_cost  * battery_capacity

                if equipment == 'inverter_life':
                    new_inverter_cost = reduced_tech_cost (start_year = start_year, reinvestment_year = reinvestment_year, tech_cost_per_unit_start_year = inverter_cost,learning_rate = annual_inverter_cost_reduction)
                    print ("new inverter cost", new_inverter_cost, "inverter size", inverter_size)
                    inv_investment_cost_array[reinvestment_year] = new_inverter_cost * inverter_size

    #Investment costs
    investment_cost_array = pv_investment_cost_array + batt_investment_cost_array +  inv_investment_cost_array + balance_of_system_investment_cost_array + dist_investment_cost_array

    #O & M costs
    annual_o_m_costs = (o_m_costs_generation_system * investment_cost_array[0]) + (o_m_costs_distribution_system * distribution_investment_cost)
    o_m_costs_array = np.full(project_life + 1, annual_o_m_costs)  # create array of annual values
    o_m_costs_array[0] = 0  # make sure initial value i

    #energy array
    energy_array = np.full(project_life + 1, total_energy_per_year) #assume same demand annually
    energy_array[0] = 0

    #Discount factor
    discount_factor = (1 + cost_of_capital)**year

    #Discounted costs
    discounted_costs = np.sum((investment_cost_array  + o_m_costs_array)/discount_factor)

    #discounted energy
    discounted_energy = np.sum(energy_array/discount_factor)

    #LCOE
    LCOE = discounted_costs/discounted_energy

    return LCOE
