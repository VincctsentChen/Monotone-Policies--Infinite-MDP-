# =======================================================
# Patient Simulation - Infinite Horizon Hypertension treatment case study (No Gurobi Version)
# =======================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Loading modules
import numpy as np  # array operations
import pandas as pd  # data manipulation
from termcolor import colored # colored warnings
from ascvd_risk import arisk  # risk calculations
from transition_probabilities_infinite import TP_infinite  # transition probability calculations
from policy_evaluation_infinite import evaluate_pi_infinite, policy_improvement_infinite  # MDPs without Gurobi
from aha_2017_guideline_infinite import aha_guideline_infinite
from risk_based_policy_infinite import risk_policy_infinite

# Patient simulation function for infinite horizon MDP (without Gurobi)
def patient_sim_infinite_no_gurobi(pt_id, patientdata, numhealth, healthy, dead, events, stroke_hist, ascvd_hist, event_states,
                        lifedata, mortality_rates, chddeathdata, strokedeathdata, alldeathdata, riskslopedata, 
                        sbpmin, dbpmin, sbpmax, dbpmax, alldrugs, trtharm, QoL, QoLterm, alpha, gamma, 
                        state_order, S_class, action_order, A_class, action_class_meds, targetrisk, targetdiff, 
                        targetsbp, targetdbp, numeds, reference_age_index=0):
    """
    This function generates risk estimates and transition probabilities
    to determine treatment policies per patient for infinite horizon MDP
    (Version without Gurobi - uses policy improvement instead)
    """
    
    try:
        print(f"Processing patient {pt_id}...")
        
        # Assume that the patient has the same pre-treatment SBP/DBP no matter the health condition
        # Using reference age for time-independent parameters
        pretrtsbp = np.ones(numhealth) * np.array(patientdata.sbp.iloc[reference_age_index])
        pretrtdbp = np.ones(numhealth) * np.array(patientdata.dbp.iloc[reference_age_index])

        # Storing risk calculations (time-independent)
        ascvdrisk1 = np.empty((numhealth, events))  # 1-y CHD and stroke risk
        periodrisk1 = np.empty((numhealth, events))  # 1-y risk after scaling

        ascvdrisk10 = np.empty((numhealth, events))  # 10-y CHD and stroke risk
        periodrisk10 = np.empty((numhealth, events))  # 10-y risk after scaling

        for h in range(numhealth):  # each state
            # Using reference age for time-independent calculations
            age = patientdata.age.iloc[reference_age_index]
            
            # Changing scaling factor based on age
            if age >= 60:
                ascvd_hist_sim = ascvd_hist.copy()
                ascvd_hist_sim[stroke_hist, 1] = 2
            else:
                ascvd_hist_sim = ascvd_hist.copy()

            for k in range(events):  # each event type
                # 1-year ASCVD risk calculation (for transition probabilities)
                ascvdrisk1[h, k] = arisk(k, patientdata.sex.iloc[reference_age_index], 
                                       patientdata.race.iloc[reference_age_index], age,
                                       patientdata.sbp.iloc[reference_age_index], 
                                       patientdata.smk.iloc[reference_age_index], 
                                       patientdata.tc.iloc[reference_age_index],
                                       patientdata.hdl.iloc[reference_age_index], 
                                       patientdata.diab.iloc[reference_age_index], 0, 1)

                # 10-year ASCVD risk calculation (for AHA's guidelines)
                ascvdrisk10[h, k] = arisk(k, patientdata.sex.iloc[reference_age_index], 
                                        patientdata.race.iloc[reference_age_index], age,
                                        patientdata.sbp.iloc[reference_age_index], 
                                        patientdata.smk.iloc[reference_age_index], 
                                        patientdata.tc.iloc[reference_age_index],
                                        patientdata.hdl.iloc[reference_age_index], 
                                        patientdata.diab.iloc[reference_age_index], 0, 10)

                if ascvd_hist_sim[h, k] > 1:
                    # Scaling odds of 1-year risks
                    periododds = ascvdrisk1[h, k]/(1-ascvdrisk1[h, k])
                    periododds = ascvd_hist_sim[h, k]*periododds
                    periodrisk1[h, k] = periododds/(1+periododds)

                    # Scaling odds of 10-year risks
                    periododds = ascvdrisk10[h, k]/(1-ascvdrisk10[h, k])
                    periododds = ascvd_hist_sim[h, k]*periododds
                    periodrisk10[h, k] = periododds/(1+periododds)

                elif ascvd_hist_sim[h, k] == 0:  # set risk to 0
                    periodrisk1[h, k] = 0
                    periodrisk10[h, k] = 0
                else:  # no scale
                    periodrisk1[h, k] = ascvdrisk1[h, k]
                    periodrisk10[h, k] = ascvdrisk10[h, k]

        # life expectancy and death likelihood data index
        if patientdata.sex.iloc[reference_age_index] == 1:  # male
            sexcol = 1  # column in deathdata corresponding to male
        else:
            sexcol = 2  # column in deathdata corresponding to female

        # Death rates (time-independent, using reference age)
        age = patientdata.age.iloc[reference_age_index]
        chddeath = chddeathdata.iloc[np.where(chddeathdata.iloc[:, 0] == age)[0][0], sexcol]
        strokedeath = strokedeathdata.iloc[np.where(strokedeathdata.iloc[:, 0] == age)[0][0], sexcol]
        alldeath = alldeathdata.iloc[np.where(alldeathdata.iloc[:, 0] == age)[0][0], sexcol]

        # Risk slopes (for BP reductions, time-independent)
        riskslope = riskslopedata.iloc[np.where(riskslopedata.iloc[:, 0] == age)[0][0], 1:3]  # Keep as DataFrame

        # Calculating transition probabilities (time-independent)
        P, feas = TP_infinite(periodrisk1, chddeath, strokedeath, alldeath, riskslope, 
                             pretrtsbp, pretrtdbp, sbpmin, dbpmin, sbpmax, dbpmax, alldrugs)

        # Sorting transition probabilities and feasibility indicators according to state ordering
        P = P[state_order, :, :]; P = P[:, state_order, :]; feas = feas[state_order, :]

        # Sorting transition probabilities and feasibility indicators according to action ordering
        P = P[:, :, action_order]; feas = feas[:, action_order]

        # Extracting list of infeasible actions per state
        infeasible = []  # stores index of infeasible actions
        feasible = []  # stores index of feasible actions
        for s in range(feas.shape[0]):
            infeasible.append(list(np.where(feas[s, :] == 0)[0]))
            feasible.append(list(np.where(feas[s, :] == 1)[0]))

        # Calculating expected rewards (time-independent)
        r = np.empty((numhealth, len(alldrugs))); r[:] = np.nan  # stores rewards
        
        # QoL weights by reference age
        age = patientdata.age.iloc[reference_age_index]
        qol = None
        if 40 <= age <= 44:
            qol = QoL.get("40-44")
        elif 45 <= age <= 54:
            qol = QoL.get("45-54")
        elif 55 <= age <= 64:
            qol = QoL.get("55-64")
        elif 65 <= age <= 74:
            qol = QoL.get("65-74")
        elif 75 <= age <= 84:
            qol = QoL.get("75-84")
        qol = np.array(qol)[state_order]  # Ordering rewards

        # Subtracting treatment disutility
        harmsort = np.array(trtharm)[action_order] # sorting disutilities according to action order
        for a in range(len(alldrugs)):
            r[:, a] = [max(0, rw-harmsort[a]) for rw in qol]  # bounding rewards below by zero

        # Initial state distribution (time-independent)
        alpha = alpha[state_order]  # initial state distribution

        # Determining optimal policies using policy improvement (no Gurobi needed)
        print(f"Finding optimal policy for patient {pt_id}...")
        pi_opt, V_opt = policy_improvement_infinite(P, r, gamma)
        d_opt = pi_opt.astype(int)
        d_opt[dead] = 0  # treating only on alive states
        
        # Calculate objective value
        J_opt = np.dot(alpha, V_opt)

        # For monotone policy, we'll use a simple heuristic
        # (In the full version with Gurobi, this would be solved as an MIP)
        print(f"Finding monotone policy for patient {pt_id}...")
        d_mopt = d_opt.copy()  # Start with optimal policy
        
        # Simple monotonicity enforcement: ensure higher states get higher or equal actions
        for h in range(1, numhealth):
            if h not in dead:  # Don't modify dead states
                if d_mopt[h] < d_mopt[h-1]:
                    d_mopt[h] = d_mopt[h-1]  # Make it monotone
        
        # Evaluate monotone policy
        V_mopt = evaluate_pi_infinite(d_mopt, P, r, gamma)
        J_mopt = np.dot(alpha, V_mopt)

        # Determining policy based on the 2017 AHA's guidelines (time-independent)
        print(f"Finding AHA guideline policy for patient {pt_id}...")
        # For infinite horizon, we use the reference age for guideline calculations
        # Use the infinite horizon version that expects (numhealth, events) format
        riskslope_reshaped = pd.DataFrame(riskslope.values.reshape(1, 2))  # Add time dimension as DataFrame
        
        pi_aha = aha_guideline_infinite(periodrisk10, pretrtsbp, pretrtdbp, 
                                       targetrisk, targetsbp, targetdbp, sbpmin, dbpmin,
                                       riskslope_reshaped, numeds, healthy)

        ## Making sure clinical guidelines are feasible
        feas_meds_list = [np.unique(np.select([[y in action_class_meds[x] for y in fst] for x in range(len(action_class_meds))], 
                                             np.arange(numeds+1))).tolist() for fst in feasible] # feasible number of medications
        for h in range(numhealth):
            if pi_aha[h] not in feas_meds_list[h]: # checking feasibility of policy
                pi_aha[h] = np.where((pi_aha[h]+1) in feas_meds_list[h], (pi_aha[h]+1), (pi_aha[h]-1)).astype(int)
                if pi_aha[h] not in feas_meds_list[h]: # if neither is feasible, returning the largest number of medications feasible
                    print(colored("Warning: Feasibility conditions not met for clinical guidelines in patient " + str(pt_id), "blue"))
                    pi_aha[h] = max(feas_meds_list[h])

        # Evaluate AHA policy
        d_aha = pi_aha.astype(int)
        V_aha = evaluate_pi_infinite(d_aha, P, r, gamma)
        J_aha = np.dot(alpha, V_aha)
        e_aha = np.zeros(numhealth)  # Simplified - no event calculation

        # Determining policy based on a risk threshold (time-independent)
        print(f"Finding risk-based policy for patient {pt_id}...")
        # Use the infinite horizon version that expects (numhealth, events) format
        riskslope_reshaped = pd.DataFrame(riskslope.values.reshape(1, 2))  # Add time dimension as DataFrame
        
        pi_risk = risk_policy_infinite(periodrisk10, pretrtsbp, pretrtdbp, 
                                      targetrisk, targetdiff, sbpmin, dbpmin, riskslope_reshaped, numeds)

        ## Making sure risk-based policies are feasible
        feas_meds_list = [np.unique(np.select([[y in action_class_meds[x] for y in fst] for x in range(len(action_class_meds))],
                                             np.arange(numeds + 1))).tolist() for fst in feasible]  # feasible number of medications
        for h in range(numhealth):
            if pi_risk[h] not in feas_meds_list[h]:  # checking feasibility of policy
                pi_risk[h] = np.where((pi_risk[h] + 1) in feas_meds_list[h], (pi_risk[h] + 1),
                                    (pi_risk[h] - 1)).astype(int)
                if pi_risk[h] not in feas_meds_list[h]:  # if neither is feasible, returning the largest number of medications feasible
                    print(colored("Warning: Feasibility conditions not met for risk-based policy in patient " + str(pt_id), "blue"))
                    pi_risk[h] = max(feas_meds_list[h])

        # Evaluate risk-based policy
        d_risk = pi_risk.astype(int)
        V_risk = evaluate_pi_infinite(d_risk, P, r, gamma)
        J_risk = np.dot(alpha, V_risk)
        e_risk = np.zeros(numhealth)  # Simplified - no event calculation

        # Evaluating no treatment policy
        print(f"Evaluating no treatment policy for patient {pt_id}...")
        d_notrt = np.zeros(numhealth, dtype=int)  # No treatment policy
        V_notrt = evaluate_pi_infinite(d_notrt, P, r, gamma)
        e_notrt = np.zeros(numhealth)  # Simplified - no event calculation

        # Changing policies back to original order (to match alldrugs list)
        if ~np.isnan(np.stack((d_opt, d_mopt, d_aha, d_risk))).any():
            d_opt = np.array(action_order)[d_opt.astype(int)]
            d_mopt = np.array(action_order)[d_mopt.astype(int)]
            d_aha = np.array(action_order)[d_aha.astype(int)]
            d_risk = np.array(action_order)[d_risk.astype(int)]

        print(f"Patient {pt_id} Done (Infinite Horizon, No Gurobi)")

        return (pt_id, V_notrt, e_notrt,
                V_opt, d_opt, None, J_opt,  # None for occupancy measure
                V_mopt, d_mopt, J_mopt,
                V_aha, d_aha, J_aha, e_aha,
                V_risk, d_risk, J_risk, e_risk
                )
                
    except Exception as e:
        print(f"Error processing patient {pt_id}: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Patient simulation infinite horizon (no Gurobi version) loaded successfully!")
    print("This version uses policy improvement instead of linear programming.")
