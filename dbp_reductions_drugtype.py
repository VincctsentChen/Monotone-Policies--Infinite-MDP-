# ==============================================================
# Estimating change in DBP from standard dose for each drug type
# ==============================================================

# Loading modules
import numpy as np

# Functions to estimate the effect of each drug on DBP
def thiazides(pretreatment):
    BPdrop = 4.4+0.11*(pretreatment-97)
    return BPdrop

def aceinhibitors(pretreatment):
    BPdrop = 4.7+0.11*(pretreatment-97)
    return BPdrop

def arb(pretreatment):
    BPdrop = 5.7+0.11*(pretreatment-97)
    return BPdrop

def calciumcb(pretreatment):
    BPdrop = 5.9+0.11*(pretreatment-97)
    return BPdrop

def betablock(pretreatment):
    BPdrop = 6.7+0.11*(pretreatment-97)
    return BPdrop

# Calculating DBP reductions for each combination
def dbp_reductions(trt, pretreatment, alldrugs):

    #    #Initializing post-treatment DBP
    #    posttreatment = pretreatment

    # Initializing DBP reduction
    dbp_reduc = 0

    # Making sure evaluated treatmnet is in a list or string format
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
            dbp_reduc = dbp_reduc+thiazides(pretreatment-dbp_reduc)
    if bb > 0:  # Reductions due to Beta-blockers
        for r in range(bb):
            #            posttreatment = posttreatment-betablock(posttreatment)
            dbp_reduc = dbp_reduc+betablock(pretreatment-dbp_reduc)
    if ace > 0:  # Reductions due to ACE inhibitors
        for r in range(ace):
            #            posttreatment = posttreatment-aceinhibitors(posttreatment)
            dbp_reduc = dbp_reduc+aceinhibitors(pretreatment-dbp_reduc)
    if a2ra > 0:  # Reductions due to Angiotensin II receptor antagonists
        for r in range(a2ra):
            #            posttreatment = posttreatment-arb(posttreatment)
            dbp_reduc = dbp_reduc+arb(pretreatment-dbp_reduc)
    if ccb > 0:  # Reductions due to Calcium channel blockers
        for r in range(ccb):
            #            posttreatment = posttreatment-calciumcb(posttreatment)
            dbp_reduc = dbp_reduc+calciumcb(pretreatment-dbp_reduc)

    return dbp_reduc  # ,posttreatment

# Calculating estimated changed in DBP from standard generic dose
def DBPreduc(pretreatment):
    BPdrop = 5.5+0.11*(pretreatment-97)

    return BPdrop

# Calculating DBP reduction from standard generic dose
def dbp_reductions_generic(trt, pretreatment):

    # reductions for standard doses
    red_1std = DBPreduc(pretreatment)
    red_2std = red_1std + DBPreduc(pretreatment - red_1std)
    red_3std = red_2std + DBPreduc(pretreatment - red_2std)
    red_4std = red_3std + DBPreduc(pretreatment - red_3std)
    red_5std = red_4std + DBPreduc(pretreatment - red_4std)

    # use interpolation for mixes of standard and half
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
