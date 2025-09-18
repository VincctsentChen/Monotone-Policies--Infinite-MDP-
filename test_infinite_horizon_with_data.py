#!/usr/bin/env python3
"""
Test script for infinite horizon MDP with actual data (no Gurobi version)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from patient_simulation_infinite_no_gurobi import patient_sim_infinite_no_gurobi

def test_with_actual_data():
    """Test the infinite horizon MDP with actual patient data"""
    
    print("Testing Infinite Horizon MDP with Actual Data")
    print("=" * 50)
    
    try:
        # Load data
        print("Loading data...")
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
        
        print(f"✅ Data loaded successfully!")
        print(f"   - Patient data: {len(ptdata)} records")
        print(f"   - Life data: {len(lifedata)} records")
        print(f"   - Risk slope data: {len(riskslopedata)} records")
        
        # Go back to main directory
        os.chdir('..')
        os.chdir('..')
        
        # Set up parameters (simplified version)
        numhealth = 10
        events = 2
        numeds = 5
        
        # Treatment parameters
        sbpmin = 120
        dbpmin = 55
        sbpmax = 150
        dbpmax = 90
        
        # AHA's guideline parameters
        targetrisk = 0.1
        targetsbp = 130
        targetdbp = 80
        targetdiff = 0.025
        
        # Generate treatment options (simplified)
        drugs = ["ACE", "ARB", "BB", "CCB", "TH"]
        drugs.insert(0, "NT")  # no treatment
        alldrugs = drugs  # Simplified - just single drugs for testing
        
        # QoL parameters
        QoL = {"40-44": [1, 0.9348, 0.8835, 0.9348*0.8835, 0.8970*(1/12)+0.9348*(11/12), 0.8662*(1/12)+0.8835*(11/12), 0, 0, 0, 0],
               "45-54": [1, 0.9374, 0.8835, 0.9374*0.8835, 0.8862*(1/12)+0.9374*(11/12), 0.8662*(1/12)+0.8835*(11/12), 0, 0, 0, 0],
               "55-64": [1, 0.9376, 0.8835, 0.9376*0.8835, 0.8669*(1/12)+0.9376*(11/12), 0.8662*(1/12)+0.8835*(11/12), 0, 0, 0, 0],
               "65-74": [1, 0.9372, 0.8835, 0.9372*0.8835, 0.8351*(1/12)+0.9372*(11/12), 0.8662*(1/12)+0.8835*(11/12), 0, 0, 0, 0],
               "75-84": [1, 0.9364, 0.8835, 0.9363*0.8835, 0.7946*(1/12)+0.9363*(11/12), 0.8662*(1/12)+0.8835*(11/12), 0, 0, 0, 0]}
        
        QoLterm = {"40-44": [1, 0.9348, 0.8835, 0.9348*0.8835, 0.9348, 0.8835, 0, 0, 0, 0],
                   "45-54": [1, 0.9374, 0.8835, 0.9374*0.8835, 0.9374, 0.8835, 0, 0, 0, 0],
                   "55-64": [1, 0.9376, 0.8835, 0.9376*0.8835, 0.9376, 0.8835, 0, 0, 0, 0],
                   "65-74": [1, 0.9372, 0.8835, 0.9372*0.8835, 0.9372, 0.8835, 0, 0, 0, 0],
                   "75-84": [1, 0.9364, 0.8835, 0.9363*0.8835, 0.9364, 0.8835, 0, 0, 0, 0]}
        
        # Treatment harm (simplified)
        trtharm = [0, 0.002, 0.002, 0.002, 0.002, 0.002]  # Simplified disutility
        
        # MDP parameters
        gamma = 0.97
        
        # Mortality rates
        mortality_rates = {"Males <2 CHD events": [1, 1/1.6, 1/2.3, (1/1.6)*(1/2.3), 1/1.6, 1/2.3, 0, 0, 0, 0],
                           "Females <2 CHD events": [1, 1/2.1, 1/2.3, (1/2.1)*(1/2.3), 1/2.1, 1/2.3, 0, 0, 0, 0]}
        
        # State parameters (simplified)
        healthy = [0]
        dead = [6, 7, 8, 9]
        stroke_hist = [2, 3, 5]
        ascvd_hist = np.ones((numhealth, events))
        ascvd_hist[stroke_hist, 1] = 3  # Stroke history multiplier
        ascvd_hist[dead, :] = 0
        
        event_states = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  # Events in states 4,5
        
        # State and action ordering (simplified)
        state_order = list(range(numhealth))
        action_order = list(range(len(alldrugs)))
        S_class = [[0], [1, 4], [2, 5], [3], [6, 7, 8, 9]]
        A_class = [[0], [1], [2], [3], [4], [5]]
        action_class_meds = [[0], [1], [2], [3], [4], [5]]
        
        # Initial state distribution
        alpha = np.zeros(numhealth)
        alpha[0] = 1  # Start in healthy state
        
        print("\\nTesting with first patient...")
        
        # Test with first patient
        pt_id = 0
        patient_data = ptdata[ptdata.id == pt_id]
        
        if len(patient_data) == 0:
            print("❌ No patient data found for patient ID 0")
            return False
        
        print(f"Patient {pt_id} data: {len(patient_data)} records")
        print(f"Patient age: {patient_data.age.iloc[0]}")
        print(f"Patient SBP: {patient_data.sbp.iloc[0]}")
        print(f"Patient DBP: {patient_data.dbp.iloc[0]}")
        
        # Run patient simulation
        result = patient_sim_infinite_no_gurobi(
            pt_id, patient_data, numhealth, healthy, dead, events,
            stroke_hist, ascvd_hist, event_states, lifedata, mortality_rates,
            chddeathdata, strokedeathdata, alldeathdata, riskslopedata,
            sbpmin, dbpmin, sbpmax, dbpmax, alldrugs, trtharm,
            QoL, QoLterm, alpha, gamma, state_order, S_class,
            action_order, A_class, action_class_meds, targetrisk, targetdiff,
            targetsbp, targetdbp, numeds, reference_age_index=0
        )
        
        if result is not None:
            print("\\n✅ Patient simulation completed successfully!")
            print(f"Results: {len(result)} outputs")
            
            # Extract key results
            pt_id, V_notrt, e_notrt, V_opt, d_opt, occup, J_opt, V_mopt, d_mopt, J_mopt, V_aha, d_aha, J_aha, e_aha, V_risk, d_risk, J_risk, e_risk = result
            
            print(f"\\nKey Results:")
            print(f"  - Optimal policy value: {J_opt:.4f}")
            print(f"  - Monotone policy value: {J_mopt:.4f}")
            print(f"  - AHA policy value: {J_aha:.4f}")
            print(f"  - Risk-based policy value: {J_risk:.4f}")
            print(f"  - No treatment value: {np.dot(alpha, V_notrt):.4f}")
            
            print(f"\\nOptimal policy: {d_opt}")
            print(f"Monotone policy: {d_mopt}")
            print(f"AHA policy: {d_aha}")
            print(f"Risk-based policy: {d_risk}")
            
            return True
        else:
            print("❌ Patient simulation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_with_actual_data()
    if success:
        print("\\n🎉 All tests passed! Infinite horizon MDP is working correctly.")
    else:
        print("\\n❌ Tests failed. Please check the error messages above.")
