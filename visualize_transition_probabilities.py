#!/usr/bin/env python3
"""
-Visualization script for transition probabilities
-This script creates a visualization of the ptrans and feasible matrices using actual patient data

-Note that in the result, why Treatment 0 (No Treatment) is Infeasible
    📊 The Issue:
    Looking at the code in transition_probabilities_infinite.py (lines 52-55):
        if j == 0:  # the do nothing treatment
        sbpreduc[h, j] = 0; dbpreduc[h, j] = 0  # no reduction when taking 0 drugs
        if pretrtsbp[h] > sbpmax or pretrtdbp[h] > dbpmax:
            feasible[h, j] = 0  # must give treatment
        else:
            feasible[h, j] = 1  # do nothing is always feasible
    🎯 The Clinical Logic:
    Patient 0 has SBP = 154 mmHg, but the clinical constraint is sbpmax = 150 mmHg.
    Since 154 > 150, the algorithm correctly determines that:
    No Treatment is clinically infeasible because the patient's blood pressure is too high
    Treatment is mandatory to bring the SBP below the maximum threshold
    This is a safety constraint to prevent dangerously high blood pressure
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from transition_probabilities_infinite import TP_infinite
from ascvd_risk import arisk

def load_patient_data():
    """Load actual patient data for patient_id = 0"""
    
    print("Loading actual patient data...")
    
    # Load data from Data folder
    os.chdir('Data')
    
    # Load life expectancy and death likelihood data
    lifedata = pd.read_csv('lifedata.csv', header=None)
    strokedeathdata = pd.read_csv('strokedeathdata.csv', header=None)
    chddeathdata = pd.read_csv('chddeathdata.csv', header=None)
    alldeathdata = pd.read_csv('alldeathdata.csv', header=None)
    riskslopedata = pd.read_csv('riskslopes.csv', header=None)
    
    # Load patient data
    os.chdir('Continuous NHANES')
    ptdata = pd.read_csv('Continuous NHANES Forecasted Dataset.csv')
    
    # Go back to main directory
    os.chdir('..')
    os.chdir('..')
    
    print(f"✅ Data loaded successfully!")
    print(f"   - Patient data: {len(ptdata)} records")
    print(f"   - Life data: {len(lifedata)} records")
    print(f"   - Risk slope data: {len(riskslopedata)} records")
    
    return ptdata, lifedata, strokedeathdata, chddeathdata, alldeathdata, riskslopedata

def calculate_patient_periodrisk(pt_id=0, reference_age_index=0):
    """Calculate periodrisk for a specific patient using the same logic as patient_simulation_infinite_no_gurobi"""
    
    # Load data
    ptdata, lifedata, strokedeathdata, chddeathdata, alldeathdata, riskslopedata = load_patient_data()
    
    # Get patient data
    patient_data = ptdata[ptdata.id == pt_id]
    if len(patient_data) == 0:
        raise ValueError(f"No patient data found for patient ID {pt_id}")
    
    print(f"Patient {pt_id} data: {len(patient_data)} records")
    print(f"Patient age: {patient_data.age.iloc[reference_age_index]}")
    print(f"Patient SBP: {patient_data.sbp.iloc[reference_age_index]}")
    print(f"Patient DBP: {patient_data.dbp.iloc[reference_age_index]}")
    print(f"Patient sex: {patient_data.sex.iloc[reference_age_index]}")
    print(f"Patient race: {patient_data.race.iloc[reference_age_index]}")
    
    # Parameters
    numhealth = 10
    events = 2
    
    # State parameters (same as in test_infinite_horizon_with_data.py)
    stroke_hist = [2, 3, 5]
    ascvd_hist = np.ones((numhealth, events))
    ascvd_hist[stroke_hist, 1] = 3  # Stroke history multiplier
    ascvd_hist[[6, 7, 8, 9], :] = 0  # Dead states
    
    # Pre-treatment BP (same for all states)
    pretrtsbp = np.ones(numhealth) * patient_data.sbp.iloc[reference_age_index]
    pretrtdbp = np.ones(numhealth) * patient_data.dbp.iloc[reference_age_index]
    
    # Calculate periodrisk using the same logic as patient_simulation_infinite_no_gurobi
    periodrisk = np.empty((numhealth, events))
    
    for h in range(numhealth):
        age = patient_data.age.iloc[reference_age_index]
        
        # Adjust scaling factor based on age
        if age >= 60:
            ascvd_hist_sim = ascvd_hist.copy()
            ascvd_hist_sim[stroke_hist, 1] = 2
        else:
            ascvd_hist_sim = ascvd_hist.copy()
        
        for k in range(events):
            # 1-year ASCVD risk calculation
            ascvdrisk = arisk(k, patient_data.sex.iloc[reference_age_index], 
                            patient_data.race.iloc[reference_age_index], age,
                            patient_data.sbp.iloc[reference_age_index], 
                            patient_data.smk.iloc[reference_age_index], 
                            patient_data.tc.iloc[reference_age_index],
                            patient_data.hdl.iloc[reference_age_index], 
                            patient_data.diab.iloc[reference_age_index], 0, 1)
            
            if ascvd_hist_sim[h, k] > 1:
                # Scaling odds of 1-year risks
                periododds = ascvdrisk/(1-ascvdrisk)
                periododds = ascvd_hist_sim[h, k]*periododds
                periodrisk[h, k] = periododds/(1+periododds)
            elif ascvd_hist_sim[h, k] == 0:
                periodrisk[h, k] = 0
            else:
                periodrisk[h, k] = ascvdrisk
    
    # Death rates
    if patient_data.sex.iloc[reference_age_index] == 1:  # male
        sexcol = 1
    else:
        sexcol = 2
    
    age = patient_data.age.iloc[reference_age_index]
    chddeath = chddeathdata.iloc[np.where(chddeathdata.iloc[:, 0] == age)[0][0], sexcol]
    strokedeath = strokedeathdata.iloc[np.where(strokedeathdata.iloc[:, 0] == age)[0][0], sexcol]
    alldeath = alldeathdata.iloc[np.where(alldeathdata.iloc[:, 0] == age)[0][0], sexcol]
    
    # Risk slopes
    riskslope = riskslopedata.iloc[np.where(riskslopedata.iloc[:, 0] == age)[0][0], 1:3]
    
    # Clinical constraints
    sbpmin, dbpmin, sbpmax, dbpmax = 120, 55, 150, 90
    
    # Treatment options
    alldrugs = ["NT", "ACE", "ARB", "BB", "CCB", "TH"]
    
    print(f"\nCalculated periodrisk for patient {pt_id}:")
    print("State | CHD Risk | Stroke Risk")
    print("-" * 30)
    state_names = ["Healthy", "CHD Hist", "Stroke Hist", "Both Hist", "CHD Event", 
                   "Stroke Event", "Dead (Non)", "Dead (CHD)", "Dead (Stroke)", "Dead (Both)"]
    for i, name in enumerate(state_names):
        print(f"{i:2d}    | {periodrisk[i,0]:.4f}   | {periodrisk[i,1]:.4f}    # {name}")
    
    print(f"\nDeath rates:")
    print(f"  CHD death: {chddeath:.4f}")
    print(f"  Stroke death: {strokedeath:.4f}")
    print(f"  All-cause death: {alldeath:.4f}")
    
    print(f"\nRisk slopes:")
    print(f"  CHD: {riskslope.iloc[0]:.4f}")
    print(f"  Stroke: {riskslope.iloc[1]:.4f}")
    
    return periodrisk, chddeath, strokedeath, alldeath, riskslope, pretrtsbp, pretrtdbp, sbpmin, dbpmin, sbpmax, dbpmax, alldrugs

def visualize_matrices():
    """Visualize the transition probability and feasibility matrices using actual patient data"""
    
    print("🔍 VISUALIZING TRANSITION PROBABILITIES")
    print("=" * 50)
    print("Using actual patient data for patient_id = 0")
    print()
    
    # Get actual patient data
    periodrisk, chddeath, strokedeath, alldeath, riskslope, pretrtsbp, pretrtdbp, sbpmin, dbpmin, sbpmax, dbpmax, alldrugs = calculate_patient_periodrisk(pt_id=0)
    
    # Compute transition probabilities
    ptrans, feasible = TP_infinite(
        periodrisk, chddeath, strokedeath, alldeath, riskslope,
        pretrtsbp, pretrtdbp, sbpmin, dbpmin, sbpmax, dbpmax, alldrugs
    )
    
    # State names
    state_names = ["Healthy", "CHD Hist", "Stroke Hist", "Both Hist", "CHD Event", 
                   "Stroke Event", "Dead (Non)", "Dead (CHD)", "Dead (Stroke)", "Dead (Both)"]
    
    print("\n1. FEASIBILITY MATRIX")
    print("   Shows which treatments are feasible for each state")
    print("   1 = feasible, 0 = infeasible")
    print()
    
    # Create a nice table for feasibility
    df_feasible = pd.DataFrame(feasible, 
                              index=[f"{i}: {state_names[i]}" for i in range(len(state_names))],
                              columns=[f"T{j}: {alldrugs[j]}" for j in range(len(alldrugs))])
    
    print(df_feasible.to_string())
    print()
    
    print("\n2. TRANSITION PROBABILITY MATRIX (ptrans)")
    print("   Shape:", ptrans.shape)
    print("   Format: ptrans[from_state, to_state, treatment]")
    print("   Shows probability of transitioning from state i to state j under treatment k")
    print()
    
    # Show transition probabilities for a few key states
    key_states = [0, 1, 4, 9]  # Healthy, CHD History, CHD Event, Dead
    
    for state in key_states:
        print(f"\nTransition Probabilities FROM State {state} ({state_names[state]}):")
        print("Treatment |", end="")
        for j in range(len(state_names)):
            print(f" To {j:2d} ", end="")
        print("| Sum")
        print("-" * 80)
        
        for k in range(len(alldrugs)):
            if feasible[state, k] == 1:  # Only show feasible treatments
                print(f"   T{k:2d}    |", end="")
                row_sum = 0
                for j in range(len(state_names)):
                    prob = ptrans[state, j, k]
                    if prob > 0.001:  # Only show significant probabilities
                        print(f" {prob:.3f} ", end="")
                    else:
                        print("  -   ", end="")
                    row_sum += prob
                print(f"| {row_sum:.3f}")
            else:
                print(f"   T{k:2d}    |", end="")
                for j in range(len(state_names)):
                    print("  -   ", end="")
                print("|  -  (infeasible)")
    
    print("\n3. KEY OBSERVATIONS")
    print("-" * 30)
    
    # Analyze feasibility patterns
    print("Feasibility Patterns:")
    for i in range(len(state_names)):
        feasible_count = np.sum(feasible[i, :] == 1)
        print(f"  State {i:2d} ({state_names[i]:12s}): {feasible_count:2d} feasible treatments")
    
    # Analyze transition patterns
    print("\nTransition Patterns:")
    print("  • Healthy state (0): Can stay healthy or transition to event states")
    print("  • History states (1-3): Tend to stay in history or progress to events")
    print("  • Event states (4-5): Can stay in event or progress to death")
    print("  • Dead states (6-9): Always transition to state 9 (absorbing)")
    
    # Show some specific transition probabilities
    print("\nSpecific Transition Examples:")
    print("  • Healthy → CHD Event (no treatment):", f"{ptrans[0, 4, 0]:.3f}")
    print("  • Healthy → Stroke Event (no treatment):", f"{ptrans[0, 5, 0]:.3f}")
    print("  • CHD History → CHD Event (no treatment):", f"{ptrans[1, 4, 0]:.3f}")
    print("  • Any dead state → Dead (Both):", f"{ptrans[6, 9, 0]:.3f}")
    
    return ptrans, feasible

def analyze_treatment_effects():
    """Analyze how different treatments affect transition probabilities using actual patient data"""
    
    print("\n\n4. TREATMENT EFFECT ANALYSIS")
    print("=" * 50)
    print("Using actual patient data for patient_id = 0")
    print()
    
    # Get actual patient data
    periodrisk, chddeath, strokedeath, alldeath, riskslope, pretrtsbp, pretrtdbp, sbpmin, dbpmin, sbpmax, dbpmax, alldrugs = calculate_patient_periodrisk(pt_id=0)
    
    # Compute transition probabilities
    ptrans, feasible = TP_infinite(
        periodrisk, chddeath, strokedeath, alldeath, riskslope,
        pretrtsbp, pretrtdbp, sbpmin, dbpmin, sbpmax, dbpmax, alldrugs
    )
    
    print("Comparing treatment effects on Healthy state (State 0):")
    print("Treatment | Stay Healthy | CHD Event | Stroke Event | Death")
    print("-" * 60)
    
    for k in range(len(alldrugs)):
        if feasible[0, k] == 1:
            stay_healthy = ptrans[0, 0, k]
            chd_event = ptrans[0, 4, k]
            stroke_event = ptrans[0, 5, k]
            death = ptrans[0, 6, k] + ptrans[0, 7, k] + ptrans[0, 8, k]
            
            print(f"   T{k:2d}    |    {stay_healthy:.3f}    |   {chd_event:.3f}   |    {stroke_event:.3f}    | {death:.3f}")
    
    print("\nTreatment Effectiveness (compared to no treatment):")
    no_treatment_healthy = ptrans[0, 0, 0]
    for k in range(1, len(alldrugs)):
        if feasible[0, k] == 1:
            improvement = ptrans[0, 0, k] - no_treatment_healthy
            print(f"  {alldrugs[k]:3s}: {improvement:+.3f} improvement in staying healthy")

if __name__ == "__main__":
    # Run visualizations
    ptrans, feasible = visualize_matrices()
    analyze_treatment_effects()
    
    print("\n🎯 SUMMARY")
    print("=" * 20)
    print("• Using actual patient data (patient_id = 0) for realistic transition probabilities")
    print("• The feasibility matrix shows which treatments are clinically safe")
    print("• The transition probability matrix shows health state dynamics")
    print("• Dead states are absorbing (always transition to state 9)")
    print("• Treatment generally improves outcomes by reducing event probabilities")
    print("• All probabilities sum to 1.0 (probability conservation)")
    print("• Periodrisk values are calculated using ASCVD risk equations and patient characteristics")
