# =====================================
# Calculating transition probabilities for infinite horizon MDP
# =====================================

# Loading modules
import numpy as np
from post_treatment_risk import new_risk
from sbp_reductions_drugtype import sbp_reductions
from dbp_reductions_drugtype import dbp_reductions

# Transition probabilities' calculation for infinite horizon MDP
def TP_infinite(periodrisk, chddeath, strokedeath, alldeath, riskslope, pretrtsbp, pretrtdbp, sbpmin, dbpmin,
                sbpmax, dbpmax, alldrugs, age_index=0):
    """
    Calculating probability of health states transitions for infinite horizon MDP
    Uses time-independent parameters based on a reference age/time point
    
    Inputs:
    periodrisk: 1-year risk of CHD and stroke (time-independent, using reference age)
    chddeath: likelihood of death given a CHD event (time-independent)
    strokedeath: likelihood of death given a stroke event (time-independent)
    alldeath: likelihood of death due to non-ASCVD events (time-independent)
    riskslope: relative risk estimates of CHD and stroke events (time-independent)
    pretrtsbp: pre-treatment SBP (time-independent)
    pretrtdbp: pre-treatment DBP (time-independent)
    sbpmin (sbpmax): Minimum (maximum) SBP allowed (clinical constraint)
    dbpmin: minimum DBP allowed (clinical constraint)
    alldrugs: treatment options being considered
    age_index: index for reference age (default 0 for first age/time point)
    """

    # Extracting parameters
    numhealth = periodrisk.shape[0]  # number of states
    events = periodrisk.shape[1]  # number of events (time dimension removed)
    numtrt = len(alldrugs)  # number of treatment choices

    # Storing feasibility indicators (time-independent)
    feasible = np.empty((numhealth, numtrt)); feasible[:] = np.nan

    # Storing risk and TP calculations (time-independent)
    risk = np.empty((numhealth, events, numtrt)); risk[:] = np.nan
    ptrans = np.zeros((numhealth, numhealth, numtrt))  # state transition probabilities

    # Storing BP reductions (time-independent)
    sbpreduc = np.empty((numhealth, numtrt))
    dbpreduc = np.empty((numhealth, numtrt))

    for j in range(numtrt):  # each treatment
        for h in range(numhealth):  # each health state
            if j == 0:  # the do nothing treatment
                sbpreduc[h, j] = 0; dbpreduc[h, j] = 0  # no reduction when taking 0 drugs
                if pretrtsbp[h] > sbpmax or pretrtdbp[h] > dbpmax:
                    feasible[h, j] = 0  # must give treatment
                else:
                    feasible[h, j] = 1  # do nothing is always feasible
            else: # prescribe >0 drugs
                sbpreduc[h, j] = sbp_reductions(j, pretrtsbp[h], alldrugs)
                dbpreduc[h, j] = dbp_reductions(j, pretrtdbp[h], alldrugs)
                newsbp = pretrtsbp[h] - sbpreduc[h, j]
                newdbp = pretrtdbp[h] - dbpreduc[h, j]
                if (newsbp < sbpmin or pretrtsbp[h] < 0) or (newdbp < dbpmin or dbpreduc[h, j] < 0):
                    feasible[h, j] = 0
                else:
                    feasible[h, j] = 1

            for k in range(events):  # each event type
                # Calculating post-treatment risks (time-independent)
                risk[h, k, j] = new_risk(sbpreduc[h, j], riskslope, periodrisk[h, k], k)

            # Health state transition probabilities: allows for both CHD and stroke in same period
            # Let Dead state dominate the transition to all others
            if h == 6 or h == 7 or h == 8 or h == 9:  # Dead
                ptrans[h, 9, j] = 1  # must stay dead
            else:  # alternate denotes the state that is default state if neither a CHD event stroke, nor death occurs
                if h == 3:  # History of CHD and Stroke
                    alternate = 3
                elif h == 4 or h == 1:  # CHD Event or History of CHD
                    alternate = 1
                elif h == 5 or h == 2:  # Stroke or History of Stroke
                    alternate = 2
                else:  # Healthy
                    alternate = 0

                quits = 0
                while quits == 0:  # compute transition probabilities
                    ptrans[h, 8, j] = min(1, strokedeath * risk[h, 1, j])  # likelihood of death from stroke
                    cumulprob = ptrans[h, 8, j]

                    ptrans[h, 7, j] = min(1, chddeath * risk[h, 0, j])  # likelihood of death from CHD event
                    if cumulprob + ptrans[h, 7, j] >= 1:
                        ptrans[h, 7, j] = 1 - cumulprob
                        break
                    cumulprob += ptrans[h, 7, j]

                    ptrans[h, 6, j] = min(1, alldeath)  # likelihood of death from non CVD cause
                    if cumulprob + ptrans[h, 6, j] >= 1:
                        ptrans[h, 6, j] = 1 - cumulprob
                        break
                    cumulprob += ptrans[h, 6, j]

                    ptrans[h, 5, j] = min(1, (1 - strokedeath) * risk[h, 1, j])  # likelihood of having stroke and surviving
                    if cumulprob + ptrans[h, 5, j] >= 1:
                        ptrans[h, 5, j] = 1 - cumulprob
                        break
                    cumulprob += ptrans[h, 5, j]

                    ptrans[h, 4, j] = min(1, (1 - chddeath) * risk[h, 0, j])  # likelihood of having CHD and surviving
                    if cumulprob + ptrans[h, 4, j] >= 1:
                        ptrans[h, 4, j] = 1 - cumulprob
                        break
                    cumulprob += ptrans[h, 4, j]

                    ptrans[h, alternate, j] = 1 - cumulprob  # otherwise, you go to the alternate state
                    break

        # Making sure that no treatment is feasible if nothing else is
        if feasible[h, :].max() == 0:
            feasible[h, 0] = 1

    return ptrans, feasible
