#Author: Energy and Technology Policy Group, ETH last edited by Churchill Agutu
# Python version 2012.2
import pandas as pd
import numpy as np
import math
from Functions_script import*
from datetime import datetime


#Start timer
start_time = datetime.now()

# Show middle end time
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
#Define directories
#loads
load_1_dict = {"load_data_directory":r"C:\Users\cagutu\Pycharm\Cost_competitiveness_off-grid\Submitted_code\input_csv\load_data.csv",
               "MG_results_directory": r'C:\Users\cagutu\Pycharm\Cost_competitiveness_off-grid\Submitted_code\sample_results\MG\MG' +  ".csv",
               "SAS_results_directory": r'C:\Users\cagutu\Pycharm\Cost_competitiveness_off-grid\Submitted_code\sample_results\SAS\SAS_combined' + ".csv",
               "weighted_results_directory": r'C:\Users\cagutu\Pycharm\Cost_competitiveness_off-grid\Submitted_code\sample_results\Weighted\weighted' + ".csv"}
#irradiation data
GHI_data_df = pd.read_csv(r"C:\Users\cagutu\Pycharm\Cost_competitiveness_off-grid\Submitted_code\input_csv\GHI_data.csv", delimiter=';')

#Define the load that you want to calculate costs for
used_data = load_1_dict

#Define the load choice based on used data directory
if used_data == load_1_dict:
    load_data = load_1_dict["load_data_directory"]
    MG_file_save_directory_name = load_1_dict["MG_results_directory"]
    SAS_file_save_directory_name = load_1_dict["SAS_results_directory"]
    weighted_file_save_directory_name = load_1_dict["weighted_results_directory"]

#Source directories
print (load_data)
data = pd.read_csv(load_data, delimiter = ';')

# General_inputs
timestep = 1  # hr
#obtain load profiles
SAS_load_profiles, MG_load_profile, num_of_loads, pearson_correlation_coefficient = obtain_load_profile(data,GHI_data_df)


# Define initial sizing conditions #minigrids
pv_capacities_MG = np.hstack((0,np.arange(0,0.1,0.01),np.arange(0,1.0,0.1),np.arange(1,10,0.2),np.arange(10,50,5))) #create array with zero generation capacity included
battery_capacities_MG = np.hstack((np.arange(0,11,1),np.arange(10,101,5),np.arange(100,150,10))) #create array with zero generation capacity included
MG_system_capacities = np.stack(np.meshgrid(pv_capacities_MG,battery_capacities_MG),-1).reshape(-1,2)
MG_max_loss_of_energy_probability = 0.1
#

# Define initial sizing conditions #standalone systems
pv_capacities_SAS = np.hstack((np.arange(0,0.1,0.01),np.arange(0.1,1.0,0.1),np.arange(1,10,0.2),np.arange(10,50,5)))  #create array with zero generation capacity included
battery_capacities_SAS = np.hstack((np.arange(0.0,2,0.05),np.arange(2,10,2),np.arange(10,150,10)))  #create array with zero generation capacity included
SAS_system_capacities = np.stack(np.meshgrid(pv_capacities_SAS,battery_capacities_SAS),-1).reshape(-1,2)
SAS_max_loss_of_energy_probability = 0.1

#required outputs
result_columns_SAS = ["PV_capacity", "Battery_capacity", "LOEP", "annual_demand"]

# SAS_LCOE_tech_mix = pd.DataFrame(columns = result_columns_SAS)
SAS_LCOE_load_mix = pd.DataFrame(columns = result_columns_SAS) #for different loads
results_merger = pd.DataFrame({"PV_capacity": [], "Battery_capacity": [], "SAS_LCOE": [],"LOEP": [],"annual_demand":[]}) #added this for the values of SAS to be added

#size MG & calculate LCOE
def MG_LCOE_calculator(load, num_of_loads,MG_system_capacities):
    """
    The function calculates the levelized cost of energy (LCOE) for a given
    mini-grid system configuration, and returns the configuration that
    provides the lowest LCOE. The function uses several sub-functions to
    obtain the required inputs for the LCOE calculation. The sub-functions
    are "inverter_specs", "pv_load", "battery_specs", "gen_capacity_sizing",
    and "distribution_sizing". The code prints the resulting feasible mini-grid
    configurations with their respective LCOEs and returns the configuration with
    the lowest LCOE that meets the specified LOEP.

    :param load:
    :param num_of_loads:
    :param MG_system_capacities:
    :return:
    """

    #define required outputs
    result_columns_MG = ["PV_capacity", "Battery_capacity", "MG_LCOE", "LOEP", "annual_demand"]
    MG_LCOE_tech_mix = pd.DataFrame(columns=result_columns_MG)


    for capacities in MG_system_capacities:

        pv_capacity_MG = capacities[0] #obtain pv capacities
        battery_capacity_MG = capacities[1] #obtain battery capacities

        #Source inputs

        inverter_eff, inverter_cost, inverter_life,annual_inv_cost_reduction = inverter_specs(inverter_eff=0.85, inverter_cost=300, annual_inv_cost_reduction=0.15, inverter_life=10)

        GHI_data = GHI_data_df['GHI'].to_numpy()

        pv_capacity, pv_hourly_load = pv_load(module_size= pv_capacity_MG, num_modules=1, Tamb=25, GHI=GHI_data, NOCT=45,MPPT=-0.0038)

        tech_type, charge_efficiency, discharge_efficiency, battery_capacity, battery_power, batt_cap_max_soc, batt_cap_min_soc, annual_batt_cost_reduction,battery_life = battery_specs(tech_type="lithium_ion",
                                                                                                                                                                                         charge_efficiency=0.95,
                                                                                                                                                                                         discharge_efficiency=0.95,
                                                                                                                                                                                         battery_capacity= battery_capacity_MG,
                                                                                                                                                                                         battery_power=0.2,
                                                                                                                                                                                         battery_max_soc=1,
                                                                                                                                                                                         battery_min_soc=0.20,
                                                                                                                                                                                         annual_batt_cost_reduction=0.18,
                                                                                                                                                                                         battery_life=10 )

        inverter_size = MG_load_profile.max() #inverter should be able to handle the peak power

        #System
        loss_of_energy_probability, energy_per_year = gen_capacity_sizing(load_profile = MG_load_profile, pv_load = pv_hourly_load, inverter_eff = inverter_eff, tech_type= tech_type,
                                                              charge_eff = charge_efficiency, discharge_eff = discharge_efficiency, batt_capacity = battery_capacity, batt_cap_max_soc = batt_cap_max_soc, batt_cap_min_soc = batt_cap_min_soc)

        #distribution infrastructure
        lv_line_length, mv_line_length, lv_line_life, mv_line_life,distribution_losses,num_of_loads,transformer_life = distribution_sizing(distance_between_loads = 0.005, distribution_losses = 0.1,num_of_loads = num_of_loads)

        #calculate MG LCOE
        get_MG_lcoe =  lcoe_calculation(start_year = 0,
                                         tech_architecture = 'MG',total_energy_per_year = energy_per_year, project_life = 10,
                                         pv_life = 25, battery_life = battery_life, inverter_life = inverter_life,
                                         num_connections = num_of_loads, cost_of_capital = 0.23,
                                         lv_line_length = lv_line_length, lv_line_life= lv_line_life  ,
                                         mv_line_length = 0, mv_line_life = mv_line_life, distribution_losses = distribution_losses,
                                         pv_capacity = pv_capacity ,
                                         battery_capacity = battery_capacity, battery_power = battery_power,
                                         inverter_size = inverter_size, num_of_transformers =0, transformer_life = transformer_life,transformer_cost = 0, pv_cost = 500,
                                         annual_pv_cost_reduction = 0, balance_of_system_cost_factor = 0.2 , battery_cost = 420,annual_batt_cost_reduction =  annual_batt_cost_reduction,
                                         inverter_cost = inverter_cost,
                                         annual_inverter_cost_reduction = annual_inv_cost_reduction,
                                         o_m_costs_generation_system = 0.02, o_m_costs_distribution_system = 0.02,
                                         lv_line_cost = 2250, mv_line_cost = 0, cost_per_connection = 150)

        results = pd.DataFrame({"PV_capacity": [pv_capacity],"Battery_capacity":[battery_capacity], "MG_LCOE":[get_MG_lcoe],"LOEP":[loss_of_energy_probability], "annual_demand":[energy_per_year]})

        final_df = pd.concat([MG_LCOE_tech_mix,results], axis = 0)
        MG_LCOE_tech_mix = final_df

    # Minimum LCOE Mini-grids
    print (MG_LCOE_tech_mix)
    feasible_MG = (MG_LCOE_tech_mix.loc[(MG_LCOE_tech_mix['LOEP'].round(decimals=5) < MG_max_loss_of_energy_probability)])  # choose all systems with LOEP < 10%
    minimum_MG_LCOE = feasible_MG .loc[feasible_MG ['MG_LCOE'] == feasible_MG ['MG_LCOE'].min() ] # select lowest LCOE
    print ("selected_mini_grid",minimum_MG_LCOE )

    return minimum_MG_LCOE

#size SAS and calculate LCOE
def SAS_LCOE_calculator(load, start_year,
                         tech_architecture, project_life,
                         GHI_factor, pv_life, battery_life, inverter_life,cost_of_capital,
                         num_modules, Tamb, NOCT, MPPT, inverter_eff,
                         charge_efficiency, discharge_efficiency, battery_power, battery_max_soc, battery_min_soc,
                         distance_between_loads, distribution_losses, num_of_transformers,
                         transformer_cost, pv_cost,
                         annual_pv_cost_reduction, balance_of_system_cost_factor, battery_cost,
                         annual_batt_cost_reduction,
                         inverter_cost,
                         annual_inverter_cost_reduction,
                         o_m_costs_generation_system, o_m_costs_distribution_system,
                         lv_line_cost, mv_line_cost, cost_per_connection):

        """

        The function calculates the LCOE for SAS. It begins by creating a data frame to store
        the LCOE for different system sizes. It then iterates through different system sizes,
        trying each size to find the optimal LCOE at the levelized cost of energy produced
        (LOEP) required.

        For each system size, the function calculates the required capacities
        of the different components, such as photovoltaic(PV) panels, batteries, and inverters,
        based on the load profile, environmental conditions, and other input parameters.
        Once the component capacities and system sizes are determined, the function calculates
        the LCOE for the given system configuration. The function returns a data frame with the
        PV and battery capacities, the LCOE, and various other parameters for each system size tested.

        #the inputs correspond to the functions in the functions script
        """

        #size for specific load using different SASs
        num_of_loads = 0 #SAS are sized individually so no need to calculate number of loads
        SAS_LCOE_tech_mix = pd.DataFrame(columns=result_columns_SAS)
        hourly_load = load.iloc[:,-1]
        total_load = load.iloc[:,-1].sum()
        print ("sum", load.iloc[:,-1].sum())

        if total_load > 0: #remove empty columns
            for capacities in SAS_system_capacities:  # try different system sizes for each load and find min LCOE at LOEP required
                pv_capacity_SA = capacities[0]  # obtain pv capacities
                battery_capacity_SA = capacities[1]  # obtain battery capacities

                inverter_eff, inverter_cost, inverter_life,annual_inv_cost_reduction = inverter_specs(inverter_eff=inverter_eff, inverter_cost= inverter_cost, annual_inv_cost_reduction= annual_inverter_cost_reduction, inverter_life=inverter_life)

                #Convert GHI series to dataframe
                GHI_data = GHI_data_df['GHI'].to_numpy() * GHI_factor

                # pv capacities
                pv_capacity, pv_hourly_load = pv_load(module_size=pv_capacity_SA, num_modules=num_modules, Tamb=Tamb, GHI=GHI_data, NOCT=NOCT, MPPT=MPPT)
                # print ("PV LOADS", pv_gen_res)

                #Battery_specs
                tech_type, charge_efficiency, discharge_efficiency, battery_capacity, battery_power, batt_cap_max_soc, batt_cap_min_soc, annual_batt_cost_reduction, battery_life = battery_specs(tech_type="lithium_ion",
                                                                                                                                                                                                  charge_efficiency=charge_efficiency,
                                                                                                                                                                                                  discharge_efficiency=discharge_efficiency,
                                                                                                                                                                                                  battery_capacity=battery_capacity_SA,
                                                                                                                                                                                                  battery_power=battery_power,
                                                                                                                                                                                                  battery_max_soc=battery_max_soc,
                                                                                                                                                                                                  battery_min_soc=battery_min_soc,
                                                                                                                                                                                                  annual_batt_cost_reduction=annual_batt_cost_reduction,
                                                                                                                                                                                                  battery_life=battery_life)
                inverter_size = load.max()  # inverter should be able to handle the peak power

                loss_of_energy_probability, energy_per_year = gen_capacity_sizing(load_profile=load,
                                                                                  pv_load=pv_hourly_load,
                                                                                  inverter_eff=inverter_eff,
                                                                                  tech_type=tech_type,
                                                                                  charge_eff=charge_efficiency,
                                                                                  discharge_eff=discharge_efficiency,
                                                                                  batt_capacity=battery_capacity,
                                                                                  batt_cap_max_soc=batt_cap_max_soc,
                                                                                  batt_cap_min_soc=batt_cap_min_soc)

                # distribution infrastructure
                lv_line_length, mv_line_length, lv_line_life, mv_line_life, distribution_losses, num_of_loads, transformer_life = distribution_sizing(distance_between_loads = distance_between_loads, distribution_losses = distribution_losses, num_of_loads = num_of_loads) # does not feature in calculation just needed for function to work

                #calculate LCOE

                get_SAS_lcoe = lcoe_calculation(start_year=start_year,
                                               tech_architecture='SAS', total_energy_per_year=energy_per_year, project_life=project_life,
                                               pv_life=pv_life, battery_life=battery_life, inverter_life=inverter_life,
                                               num_connections=num_of_loads, cost_of_capital = cost_of_capital,
                                               lv_line_length=lv_line_length, lv_line_life=lv_line_life, mv_line_length=mv_line_length,
                                               mv_line_life=mv_line_life, distribution_losses=distribution_losses,
                                               pv_capacity=pv_capacity,
                                               battery_capacity=battery_capacity, battery_power=battery_power,
                                               inverter_size=inverter_size, num_of_transformers=num_of_transformers,
                                               transformer_life=transformer_life, transformer_cost=transformer_cost, pv_cost= pv_cost,
                                               annual_pv_cost_reduction=annual_pv_cost_reduction, balance_of_system_cost_factor=balance_of_system_cost_factor, battery_cost=battery_cost,
                                               annual_batt_cost_reduction=annual_batt_cost_reduction,
                                               inverter_cost=inverter_cost,
                                               annual_inverter_cost_reduction=annual_inv_cost_reduction,
                                               o_m_costs_generation_system=o_m_costs_generation_system, o_m_costs_distribution_system=o_m_costs_distribution_system,
                                               lv_line_cost=lv_line_cost, mv_line_cost=mv_line_cost, cost_per_connection=cost_per_connection)

                results = pd.DataFrame({"PV_capacity": [pv_capacity], "Battery_capacity": [battery_capacity], "SAS_LCOE": [get_SAS_lcoe], "LOEP": [loss_of_energy_probability], "annual_demand":[energy_per_year]})
                # results = pd.DataFrame({"PV_capacity": [pv_capacity], "Battery_capacity": [battery_capacity], "SAS_LCOE": [get_SAS_lcoe],"LOEP": [round(loss_of_energy_probability, 1)], "annual_demand": [energy_per_year]})
                final_df = pd.concat([SAS_LCOE_tech_mix, results], axis=0)
                SAS_LCOE_tech_mix = final_df

            # select the minimum SAS for this specific load

            feasible_SAS = (SAS_LCOE_tech_mix.loc[(SAS_LCOE_tech_mix['LOEP'].round(decimals=5) <= SAS_max_loss_of_energy_probability)])  # round up because e.g 0.12 is still more or less workable for a 0.1

            minimum_SAS_LCOE = feasible_SAS.loc[feasible_SAS['SAS_LCOE'] == feasible_SAS['SAS_LCOE'].min()]  # if you have many similar LOEP, select lowest LCOE
            SAS_LCOE_load_mix = minimum_SAS_LCOE  # 0R IF ALL ARE ZEROS THEN JUST TAKE THE SMALLEST LCOE, CORRELATION ANALYSIS ,DEBUG SOME VALUES ARE CALCULATING AN LOEP GREATER THAN 1 WHICH IS NOT TRUE, NEED TO FIND OUT HOW TECH INPUTS DIFFER WITH APPROACH SAS VS MG# highest minimum LOEP values
            print("SAS BEST PER APPROACH by design", SAS_LCOE_load_mix)


            new_SAS_LCOE = SAS_LCOE_load_mix['SAS_LCOE'][0]
            print ("NEW LCOE", new_SAS_LCOE)


            #updated dataframe
            SAS_LCOE_load_mix_final = pd.DataFrame({"PV_capacity":SAS_LCOE_load_mix ['PV_capacity'], "Battery_capacity": SAS_LCOE_load_mix['Battery_capacity'],"SAS_LCOE": [new_SAS_LCOE], "LOEP": SAS_LCOE_load_mix['LOEP'], "annual_demand":SAS_LCOE_load_mix ['annual_demand']})

            print("SAS BEST PER APPROACH", SAS_LCOE_load_mix_final)


            return SAS_LCOE_load_mix_final


# MGLCOE function run
# #
print ("calculating MG LCOE...")
MG_LCOE_calculation = MG_LCOE_calculator(load=MG_load_profile, num_of_loads=num_of_loads, MG_system_capacities=MG_system_capacities)

# # # create final name
MG_file_name = MG_file_save_directory_name
# # # save MG combined_data
#
MG_LCOE_calculation.to_csv(MG_file_name)
# #
print ("calculating SAS LCOE...")
# SAS LCOE function run for all loads
for index,load in enumerate(SAS_load_profiles):

    # print ("loads",load)
    sas_lcoes = SAS_LCOE_calculator(load, start_year=0,
             tech_architecture='SAS', project_life=10, GHI_factor = 1, inverter_eff = 0.85,
             pv_life=25, battery_life=10, inverter_life=10,cost_of_capital=0.17,
             num_modules=1, Tamb=25, NOCT=45, MPPT=-0.0038,
             charge_efficiency=0.95, discharge_efficiency=0.95, battery_power=0.2, battery_max_soc=1, battery_min_soc=0.20,
             distance_between_loads= 0, distribution_losses=0, num_of_transformers=0,
             transformer_cost=0, pv_cost=500,
             annual_pv_cost_reduction=0, balance_of_system_cost_factor= 0.2, battery_cost=420,
             annual_batt_cost_reduction=0.18,
             inverter_cost=300,
             annual_inverter_cost_reduction=0.15,
             o_m_costs_generation_system=0.02, o_m_costs_distribution_system=0.02,
             lv_line_cost=0, mv_line_cost=0, cost_per_connection=0)

    # obtain for all loads
    SAS_combined = pd.concat([sas_lcoes, results_merger]) #append the chossen value to the list
    results_merger = SAS_combined #update dictionary with new values
#
#
 #create final name
SAS_combined_file_name = SAS_file_save_directory_name
#save SAS combined_data
SAS_combined.to_csv(SAS_combined_file_name)
# # # # # #weighted SAS average
SAS_LCOE_weighted_average = (SAS_combined['SAS_LCOE'] * SAS_combined['annual_demand']).sum()/SAS_combined['annual_demand'].sum()

# convert weighted average to csv
final_results = pd.DataFrame({"Weighted_SAS_LCOE": [SAS_LCOE_weighted_average]})
# #save_final results
final_results.to_csv(weighted_file_save_directory_name)

# Show end time
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))