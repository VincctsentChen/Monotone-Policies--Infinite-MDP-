# =========================
# Calculating ASCVD risk
# ========================

def arisk(event, sex, race, age, sbp, smk, tc, hdl, diab, trt, time):
    # ASCVD risk calculator (2013 ACC/AHA Guideline)
    # inputs: type of risk to calculate (0=CHD, 1=stroke), sex (1=male, 0=female),
    # race (1=white,0=black), age, SBP, smoking status (1=smoker, 0=nonsmoker), total
    # cholesterol, HDL, diabetes status (1=diabetic, 0=nondiabetic), trt
    # (1=BP reported is on treatment, 0=BP reported is untreated), time.
    # outputs: likelihood of CHD or stroke in the next "time" years

    import math
    import sys

    if sex == 1:  # male
        if race == 1:  # white
            b_age = 12.344
            b_age2 = 0
            b_tc = 11.853
            b_age_tc = -2.664
            b_hdl = -7.990
            b_age_hdl = 1.769

            if trt == 1:  # SBP is treated SBP
                b_sbp = 1.797
                b_age_sbp = 0
            else:  # SBP is untreated SBP
                b_sbp = 1.764
                b_age_sbp = 0

            b_smk = 7.837
            b_age_smk = -1.795
            b_diab = 0.658
            meanz = 61.18

            if time == 1:
                basesurv = 0.99358
            elif time == 5:
                basesurv = 0.96254
            elif time == 10:
                basesurv = 0.9144
        #      else:
        #        stop(cat(time, "is an improper time length for risk calculation"))

        else:  # black
            b_age = 2.469
            b_age2 = 0
            b_tc = 0.302
            b_age_tc = 0
            b_hdl = -0.307
            b_age_hdl = 0

            if trt == 1:  # SBP is treated SBP
                b_sbp = 1.916
                b_age_sbp = 0
            else:  # SBP is untreated SBP
                b_sbp = 1.809
                b_age_sbp = 0

            b_smk = 0.549
            b_age_smk = 0
            b_diab = 0.645
            meanz = 19.54

            if time == 1:
                basesurv = 0.99066
            elif time == 5:
                basesurv = 0.95726
            elif time == 10:
                basesurv = 0.8954
            else:
                sys.exit(str(time)+" is an improper time length for risk calculation")

    else:  # female
        if race == 1:  # white
            b_age = -29.799
            b_age2 = 4.884
            b_tc = 13.540
            b_age_tc = -3.114
            b_hdl = -13.578
            b_age_hdl = 3.149

            if trt == 1:  # SBP is treated SBP
                b_sbp = 2.019
                b_age_sbp = 0
            else:  # SBP is untreated SBP
                b_sbp = 1.957
                b_age_sbp = 0

            b_smk = 7.574
            b_age_smk = -1.665
            b_diab = 0.661
            meanz = -29.18

            if time == 1:
                basesurv = 0.99828
            elif time == 5:
                basesurv = 0.98898
            elif time == 10:
                basesurv = 0.9665
            else:
                sys.exit(str(time)+" is an improper time length for risk calculation")

        else:  # black
            b_age = 17.114
            b_age2 = 0
            b_tc = 0.940
            b_age_tc = 0
            b_hdl = -18.920
            b_age_hdl = 4.475

            if trt == 1:  # SBP is treated SBP
                b_sbp = 29.291
                b_age_sbp = -6.432
            else:  # SBP is untreated SBP
                b_sbp = 27.820
                b_age_sbp = -6.087

            b_smk = 0.691
            b_age_smk = 0
            b_diab = 0.874
            meanz = 86.61

            if time == 1:
                basesurv = 0.99834
            elif time == 5:
                basesurv = 0.98194
            elif time == 10:
                basesurv = 0.9533
            else:
                sys.exit(str(time)+" is an improper time length for risk calculation")

    # proportion of ascvd assumed to be CHD or stroke, respectively
    eventprop = [0.6, 0.4]

    indivz = b_age*math.log(age)+b_age2*(math.log(age))**2+b_tc*math.log(tc)+b_age_tc*math.log(age)*math.log(
        tc)+b_hdl*math.log(hdl)+b_age_hdl*math.log(age)*math.log(hdl)+b_sbp*math.log(sbp)+b_age_sbp*math.log(
        age)*math.log(sbp)+b_smk*smk+b_age_smk*math.log(age)*smk+b_diab*diab

    risk = eventprop[event]*(1-basesurv**(math.exp(indivz-meanz)))

    return risk
