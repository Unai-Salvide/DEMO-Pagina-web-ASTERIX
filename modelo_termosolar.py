import pypsa
import pandas as pd
import numpy as np
import numpy_financial as npf
import logging

# Forzar a Pandas a utilizar el sistema de strings clásico (NumPy) en lugar de PyArrow
# Esto evita el error "Invalid array type: ArrowStringArray" en xarray/PyPSA
pd.options.future.infer_string = False
try:
    pd.options.mode.string_storage = "python"
except Exception:
    pass

# Silenciar logs para no saturar la consola del servidor
logging.getLogger("pypsa").setLevel(logging.WARNING)
logging.getLogger("linopy").setLevel(logging.WARNING)

# --- Funciones Auxiliares ---
def calculate_financial_metrics(total_income, capex, operation_maintenance_factor=0.015, real_discount_rate=0.06, total_project_life=30) -> tuple:
    annual_cash_flow = total_income - capex * operation_maintenance_factor
    cash_flows = [- capex] + [annual_cash_flow] * total_project_life
    res_npv = npf.npv(rate=real_discount_rate, values=cash_flows)
    try:
        res_irr = npf.irr(cash_flows)
    except (ValueError, TypeError):
        res_irr = None

    cumulative = -capex
    prev_cum = cumulative
    res_dpp = None
    for year in range(1, total_project_life + 1):
        discounted_cf = annual_cash_flow / (1 + real_discount_rate) ** year
        cumulative += discounted_cf
        if cumulative >= 0:
            frac = (0 - prev_cum) / (cumulative - prev_cum)
            res_dpp = (year - 1) + frac
            break
        prev_cum = cumulative
    return res_npv, res_irr, res_dpp

def calculate_IPH(dischargingPower, apertureArea, chargingPower):
    chargingIPHpower = 0.5857 * chargingPower + 0.3284
    atmHRSGIPHpower = 0.0000465 * apertureArea - 0.0151629
    dischargingIPHpower = 0.1108886 * dischargingPower - 0.0016426 if dischargingPower < 20 else 0.0713415 * dischargingPower + 0.0910533
    totalChargingIPHpower = chargingIPHpower
    totalDischargingIPHpower = atmHRSGIPHpower + dischargingIPHpower
    nominalIPHpower = max(totalChargingIPHpower, totalDischargingIPHpower)
    return totalChargingIPHpower, totalDischargingIPHpower, nominalIPHpower

def calculate_capex_largeScale(dischargingPower, chargingPower, tesHours, reservoirVolume, apertureArea, nIPH, pOthers):
    updatedCEPCI = 796.3
    eff_cycle = 0.50
    turbinesMassFlow = 1.405 * dischargingPower + 0.4226
    compressorMassFlow = 1.8361 * chargingPower + 0.2517
    gasturbinesPower = 0.9 * dischargingPower
    bottomingcyclePower = 0.1 * dischargingPower

    turbineHPT = (380.46*turbinesMassFlow*(1/(0.92-0.9033))*(np.log(40/8))*(1+(np.exp(0.036*(273.15 + 550-1511)))))*(updatedCEPCI/575.4)
    turbineLPT = (380.46*turbinesMassFlow*(1/(0.92-0.875))*(np.log(8/1.013))*(1+(np.exp(0.036*(273.15 + 750-1511)))))*(updatedCEPCI/575.4)
    generatorAndGearBox = (20000*((gasturbinesPower*1000)/100)**0.6)*(updatedCEPCI/796.3) * 1.4
    bottomingCycle = bottomingcyclePower * 1 * (updatedCEPCI/567.5) if dischargingPower < 20 else (((bottomingcyclePower/0.26)*1000)*218.44 + 4000000) * 0.67 * (updatedCEPCI/567.3) + (8220*(bottomingcyclePower*1000)**0.7) * (updatedCEPCI/603.1)
    compressorTrain = ( 4 * (1.051*(39.5*compressorMassFlow/(0.9-0.84))*((80/1.013)**(1/4))*np.log((80/1.013)**(1/4))))*(updatedCEPCI/394.3)
    airReservoir = reservoirVolume * 6700 * (updatedCEPCI/796.3) if dischargingPower < 20 else (reservoirVolume*62 + 37500000) * (updatedCEPCI/603.1)
    heatExchangers = ( 8500000*((turbinesMassFlow/30)**0.7) + 4000000*(((compressorMassFlow/15))**0.7) ) * (updatedCEPCI/796.3)
    thermalTES = ((dischargingPower / eff_cycle) * tesHours * 1000) * 20 * (updatedCEPCI/567.5)
    solarField = apertureArea * 75 * (updatedCEPCI/567.5)
    receiver = (0.0016 * apertureArea + 17.376) * 35000 * (updatedCEPCI/468.2)
    tower = (21.09 * apertureArea + 1029169.92 ) * (updatedCEPCI/796.3)
    blower = (3.23 * apertureArea + 34784.35) * (updatedCEPCI/796.3)
    iph = (nIPH * 1000 * 120 +  593 * 2000 * max(compressorMassFlow/15, turbinesMassFlow/30) + nIPH * 1000 * 6 * 16.5) * (updatedCEPCI/796.3)

    others = ((turbineHPT + turbineLPT + generatorAndGearBox + bottomingCycle + compressorTrain + airReservoir + heatExchangers + thermalTES + solarField + receiver + tower + blower + iph) * pOthers)  / (1-pOthers)
    total_capex = turbineHPT + turbineLPT + generatorAndGearBox + bottomingCycle + compressorTrain + airReservoir + heatExchangers + thermalTES + solarField + receiver + tower + blower + iph + others
    return total_capex

def extra_functionality(n, snapshots):
    m = n.model
    status = m.add_variables(coords=[snapshots], name="expander_active", binary=True)
    
    # MODIFICACIÓN: Cambiar Link= y Generator= por name=
    p_expander = n.model.variables['Link-p'].sel(name='Expander_Turbine')
    p_compressor = n.model.variables['Link-p'].sel(name='Compressor')
    p_solar = n.model.variables['Generator-p'].sel(name='CSP_Source')
    
    exp_nom = n.links.at['Expander_Turbine', 'p_nom']
    comp_nom = n.links.at['Compressor', 'p_nom']
    sol_nom = n.generators.at['CSP_Source', 'p_nom']

    m.add_constraints(p_expander - (status * exp_nom) <= 0, name="expander_logic")
    m.add_constraints(p_compressor + (status * comp_nom) <= comp_nom, name="compressor_logic")
    m.add_constraints(p_solar + (status * sol_nom) <= sol_nom, name="solar_inflow_logic")


# --- Función Principal a exportar ---
def simular_planta(power_rating, TES_hours, reservoir_volume, aperture_area, pais, charging_power=67):
    """
    Ejecuta la optimización PyPSA con los parámetros dados y devuelve un diccionario con los resultados.
    """
    window = 24 * 31
    step = 24 * 30
    p_min, p_max = 40, 80
    mw_to_kgh = 6600
    kgh_to_mw = 1.975e-4
    eff_cycle = 0.50
    eff_solar_to_thermal = -0.02026 * np.log(aperture_area) + 0.79844
    eff_tes_storage = 0.97
    subsides = 0
    others_percentage = 0.3
    iph_heat_price = 50

    m_min = (p_min * 1e5 * reservoir_volume) / (287.05 * 303.15)
    m_max = (p_max * 1e5 * reservoir_volume) / (287.05 * 303.15)
    mass_nominal_capacity = m_max - m_min

    electricity_file_path = f'{pais}.txt'
    solar_resource_file_path = f'TMY{pais}.txt'
    
    n = pypsa.Network()
    snapshots = pd.date_range("2025-01-01 00:00", "2025-12-31 23:00", freq="h")
    n.set_snapshots(snapshots)

    n.add("Carrier", "AC")
    n.add("Carrier", "compressed_air")
    n.add("Carrier", "thermal")
    n.add("Bus", "Grid", carrier="AC")
    n.add("Bus", "Air_Bus", carrier="compressed_air")
    n.add("Bus", "Thermal_Bus", carrier="thermal")

    price_data = np.loadtxt(electricity_file_path, skiprows=2)
    price_series = price_data[:len(n.snapshots), 1]
    
    solar_data = np.loadtxt(solar_resource_file_path, skiprows=5)
    solar_series = solar_data[:len(n.snapshots), 1]
    available_solar_thermal = solar_series * aperture_area * eff_solar_to_thermal / 1e6

    n.add("Generator", "Market", bus="Grid", marginal_cost=price_series, p_nom_extendable=True, p_min_pu=-(charging_power/power_rating))
    n.add("Generator", "CSP_Source", bus="Thermal_Bus", p_nom=available_solar_thermal.max(), p_max_pu=available_solar_thermal / available_solar_thermal.max() if available_solar_thermal.max() > 0 else 0, marginal_cost=0)
    n.add("StorageUnit", "TES_Battery", bus="Thermal_Bus", p_nom=max(power_rating / eff_cycle, available_solar_thermal.max()), max_hours=TES_hours * (power_rating / eff_cycle) / max(power_rating / eff_cycle, available_solar_thermal.max()), efficiency_store=eff_tes_storage, standing_loss=0.001)
    n.add("Store", "Air_Reservoir", bus="Air_Bus", e_nom=mass_nominal_capacity, e_min_pu=0)
    n.add("Link", "Compressor", bus0="Grid", bus1="Air_Bus", p_nom=charging_power, efficiency=mw_to_kgh, efficiency2=0, p_min_pu=0)
    n.add("Link", "Expander_Turbine", bus0="Air_Bus", bus1="Grid", bus2="Thermal_Bus", p_nom=power_rating / kgh_to_mw, efficiency=kgh_to_mw, efficiency2= - (kgh_to_mw / eff_cycle))

    total_hours = len(n.snapshots)    
    final_p_discharged = pd.Series(0.0, index=n.snapshots)
    final_p_charged = pd.Series(0.0, index=n.snapshots)
    final_soc_thermal = pd.Series(0.0, index=n.snapshots)
    final_soc_air = pd.Series(0.0, index=n.snapshots)
    final_solar_captured = pd.Series(0.0, index=n.snapshots)
    final_thermalTES_in = pd.Series(0.0, index=n.snapshots)

    current_soc_air = mass_nominal_capacity * 0.5
    current_soc_thermal = 0.5

    for start in range(0, total_hours, step):
        end = min(start + window, total_hours)
        window_snaps = n.snapshots[start:end]
        commit_snaps = n.snapshots[start:min(start + step, total_hours)]
        
        n.stores.at["Air_Reservoir", "e_initial"] = current_soc_air
        n.storage_units.at["TES_Battery", "state_of_charge_initial"] = current_soc_thermal
        
        status = n.optimize(window_snaps, solver_name='highs', extra_functionality=extra_functionality, solver_options={'threads': 1, 'mip_rel_gap': 0.01}, io_api='direct', log_level='WARNING', consistency_check=False)
        
        final_p_discharged.loc[commit_snaps] = abs(n.links_t.p1["Expander_Turbine"].loc[commit_snaps])
        final_p_charged.loc[commit_snaps] = abs(n.links_t.p0["Compressor"].loc[commit_snaps])
        final_soc_thermal.loc[commit_snaps] = n.storage_units_t.state_of_charge["TES_Battery"].loc[commit_snaps]
        final_soc_air.loc[commit_snaps] = n.stores_t.e["Air_Reservoir"].loc[commit_snaps]
        final_solar_captured.loc[commit_snaps] = n.generators_t.p["CSP_Source"].loc[commit_snaps]
        final_thermalTES_in.loc[commit_snaps] = n.storage_units_t.p["TES_Battery"].loc[commit_snaps].clip(upper=0).abs()
        
        current_soc_air = final_soc_air.loc[commit_snaps[-1]]
        current_soc_thermal = final_soc_thermal.loc[commit_snaps[-1]]

    ch_IPH, disch_IPH, nominal_IPH = calculate_IPH(power_rating, aperture_area, charging_power) if calculate_IPH(power_rating, aperture_area, charging_power)[2] < 40 else (0, 0, 0)
    cc_plant = calculate_capex_largeScale(power_rating, charging_power, TES_hours, reservoir_volume, aperture_area, nominal_IPH, others_percentage)
    total_project_capex = cc_plant * (1 - subsides/100)

    total_energy_charged_grid = final_p_charged.sum()
    total_energy_discharged_grid = final_p_discharged.sum()
    hourlyIPH = np.where(final_p_charged > 0, ch_IPH * (final_p_charged/charging_power), 0) + np.where(final_p_discharged > 0, disch_IPH * (final_p_discharged/power_rating), 0)

    revenue = (final_p_discharged * price_series).sum()
    charging_cost = (final_p_charged * price_series).sum()
    totalrevenueIPH = (hourlyIPH * iph_heat_price).sum()
    operational_profit = revenue + totalrevenueIPH - charging_cost

    erte_val = (total_energy_discharged_grid / total_energy_charged_grid * 100) if total_energy_charged_grid > 1e-3 else 0
    capacity_factor = total_energy_discharged_grid * 100 / (power_rating * len(n.snapshots))

    npv, irr, dpp = calculate_financial_metrics(operational_profit, total_project_capex)
    
    tes_nom_cap = (power_rating / eff_cycle) * TES_hours

    df_results = pd.DataFrame({
        "Price": price_series,
        "P_Disch": final_p_discharged,
        "P_Ch": -final_p_charged,
        "SoC_Thermal_%": (final_soc_thermal / tes_nom_cap * 100),
        "SoC_Air_%": (final_soc_air / mass_nominal_capacity * 100),
    })

    return {
        "status": status,
        "capex_meur": total_project_capex * 1e-6,
        "npv_meur": (npv * 1e-6) if npv is not None else 0,
        "irr_pct": (irr * 100) if irr is not None else 0,
        "dpp_years": dpp if dpp is not None else 0,
        "profit_eur": operational_profit,
        "erte_pct": erte_val,
        "cf_pct": capacity_factor,
        "df_hourly": df_results
    }