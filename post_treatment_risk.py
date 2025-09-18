# =========================================================
# Calculating relative risk reductions based on risk slopes
# =========================================================

def new_risk(sbpreduc, riskslope, pretrtrisk, event):

    # post trt risk for each event type
    # event (0=CHD, 1=stroke)

    RR = (list(riskslope)[event])**(sbpreduc/20)
    risk = RR*pretrtrisk

    return risk
