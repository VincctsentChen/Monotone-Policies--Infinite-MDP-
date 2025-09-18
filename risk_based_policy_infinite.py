# ==================================================
# Risk-based policy (Infinite Horizon Version)
# ==================================================

# Loading modules
import numpy as np
from post_treatment_risk import new_risk
from sbp_reductions_drugtype import sbp_reductions_generic
from dbp_reductions_drugtype import dbp_reductions_generic

# Function to obtain policy according to risk-based guidelines (Infinite Horizon Version)
def risk_policy_infinite(pretrtrisk, pretrtsbp, pretrtdbp, targetrisk, targetdiff, sbpmin, dbpmin, riskslope, numtrt):
    """
    Infinite horizon version of risk-based policy - time-independent policy
    """
    
    # Extracting parameters
    numhealth = pretrtrisk.shape[0]  # number of states
    
    # Array to store results (initializing with no treatment)
    policy = np.empty(numhealth); policy[:] = np.nan
    
    # Determining action per health state (time-independent)
    for h in range(numhealth):  # each health state
        
        # Start with no treatment
        past_trt = 0
        
        # Calculating post-treatment risk and BP with past treatment
        sbpreduc = sbp_reductions_generic(past_trt, pretrtsbp[h])
        
        post_trt_risk = new_risk(sbpreduc, riskslope.iloc[0, :], pretrtrisk[h, 0], 0) +\
                        new_risk(sbpreduc, riskslope.iloc[0, :], pretrtrisk[h, 1], 1)
        
        # Making sure that BP is not on target without increasing treatment
        if post_trt_risk >= (targetrisk-targetdiff): # Moderate risk for ASCVD
            
            # Simulating 1-month evaluations within each year
            month = 1  # initial month
            while month <= 12 and post_trt_risk >= (targetrisk-targetdiff):  # risk not on target with current medication within the same year
                
                # Attempting to increase treatment
                if (past_trt + 1) > numtrt:
                    new_trt = past_trt # cannot give more than 5 medications
                else:
                    new_trt = past_trt + 1 # increase medication intensity
                
                # Calculating post-treatment BP with new potential treatment
                sbpreduc = sbp_reductions_generic(new_trt, pretrtsbp[h])
                post_trt_sbp = pretrtsbp[h] - sbpreduc
                dbpreduc = dbp_reductions_generic(new_trt, pretrtdbp[h])
                post_trt_dbp = pretrtdbp[h] - dbpreduc
                
                # Evaluating the feasibility of new treatment
                if (post_trt_sbp < sbpmin or sbpreduc < 0) or (post_trt_dbp < dbpmin or dbpreduc < 0):
                    policy[h] = past_trt # new treatment is not feasible
                else:
                    policy[h] = new_trt  # new treatment is feasible
                
                past_trt = policy[h] # next month's evaluation
                month += 1 # next month's evaluation
        else: # ASCVD risk already on target keeping past year's treatment
            policy[h] = past_trt # keep current treatment
    
    return policy
