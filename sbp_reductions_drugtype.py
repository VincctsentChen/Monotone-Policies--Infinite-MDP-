# ==============================================================
# Estimating change in SBP from standard dose for each drug type
# ==============================================================

# Loading modules
import numpy as np

# Functions to estimate the effect of each drug on SBP
def aceinhibitors(pretreatment):
    BPdrop = 8.5+0.1*(pretreatment-154)
    return BPdrop

def calciumcb(pretreatment):
    BPdrop = 8.8+0.1*(pretreatment-154)
    return BPdrop

def thiazides(pretreatment):
    BPdrop = 8.8+0.1*(pretreatment-154)
    return BPdrop

def betablock(pretreatment):
    BPdrop = 9.2+0.1*(pretreatment-154)
    return BPdrop

def arb(pretreatment):
    BPdrop = 10.3+0.1*(pretreatment-154)
    return BPdrop

# Calculating SBP reductions for each combination
def sbp_reductions(trt, pretreatment, alldrugs):

    #    #Initializing post-treatment SBP
    #    posttreatment = pretreatment

    # Initializing SBP reduction
    sbp_reduc = 0

    # Making sure evaluated treatment is in a list or string format
    if type(alldrugs[trt]) == str or type(alldrugs[trt]) == list:
        drugcomb = alldrugs[trt]
    else:
        drugcomb = list(alldrugs[trt])

    # Counting number of times a drug is given
    th = drugcomb.count('TH')
    bb = drugcomb.count('BB')
    ace = drugcomb.count('ACE')
    a2ra = drugcomb.count('ARB')
    ccb = drugcomb.count('CCB')

    if th > 0:  # Reductions due to Thiazides
        for r in range(th):
            #            posttreatment = posttreatment-thiazides(posttreatment)
            sbp_reduc = sbp_reduc+thiazides(pretreatment-sbp_reduc)
    if bb > 0:  # Reductions due to Beta-blockers
        for r in range(bb):
            #            posttreatment = posttreatment-betablock(posttreatment)
            sbp_reduc = sbp_reduc+betablock(pretreatment-sbp_reduc)
    if ace > 0:  # Reductions due to ACE inhibitors
        for r in range(ace):
            #            posttreatment = posttreatment-aceinhibitors(posttreatment)
            sbp_reduc = sbp_reduc+aceinhibitors(pretreatment-sbp_reduc)
    if a2ra > 0:  # Reductions due to Angiotensin II receptor antagonists
        for r in range(a2ra):
            #            posttreatment = posttreatment-arb(posttreatment)
            sbp_reduc = sbp_reduc+arb(pretreatment-sbp_reduc)
    if ccb > 0:  # Reductions due to Calcium channel blockers
        for r in range(ccb):
            #            posttreatment = posttreatment-calciumcb(posttreatment)
            sbp_reduc = sbp_reduc+calciumcb(pretreatment-sbp_reduc)

    return sbp_reduc  # ,posttreatment

# Calculate estimated changed in SBP from standard generic dose
def SBPreduc(pretreatment):
    BPdrop = 9.1+0.1*(pretreatment-154)
    return BPdrop

# Calculating SBP reductions from standard generic dose
def sbp_reductions_generic(trt, pretreatment):

    # reductions for standard doses
    red_1std = SBPreduc(pretreatment)
    red_2std = red_1std + SBPreduc(pretreatment - red_1std)
    red_3std = red_2std + SBPreduc(pretreatment - red_2std)
    red_4std = red_3std + SBPreduc(pretreatment - red_3std)
    red_5std = red_4std + SBPreduc(pretreatment - red_4std)

    # Identifying reduction
    if trt == 0:  # no trt
        reduction = 0
    elif trt == 1:  # 1 std
        reduction = red_1std
    elif trt == 2:  # 2 std
        reduction = red_2std
    elif trt == 3:  # 3 std
        reduction = red_3std
    elif trt == 4:  # 4 std
        reduction = red_4std
    elif trt == 5:  # 5 std
        reduction = red_5std
    else:
        reduction = np.nan

    return reduction
