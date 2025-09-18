# *************************************************
# Optimal Ranges - Continuous NHANES Extraction
# *************************************************

# Setup -------------------------------------------------------------------

rm(list = ls()[!(ls() %in% c())]) #Removing all variables

#Selecting home directory
if(Sys.info()["nodename"]=="IOE-TESLA"){ #selecting path by computer name
  main_dir = "~/Optimal Ranges"
  home_dir = "~/Optimal Ranges/Data/Continuous NHANES"
  setwd(home_dir)
}else{
  main_dir = file.path(Sys.getenv("USERPROFILE"),"/My Drive/Research/Current Projects/Optimal Ranges/Python")
  home_dir = file.path(Sys.getenv("USERPROFILE"),"/My Drive/Research/Current Projects/Optimal Ranges/Data/Continuous NHANES")
  setwd(home_dir)
}

#Loading packages (and installing if necessary)
if(!("foreign" %in% installed.packages()[,"Package"])) install.packages("foreign"); library(foreign) #read SAS files
if(!("stringr" %in% installed.packages()[,"Package"])) install.packages("stringr"); library(stringr) #split columns
if(!("doParallel" %in% installed.packages()[,"Package"])) install.packages("doParallel"); library(doParallel) #parallel computation
if(!("missForest" %in% installed.packages()[,"Package"])) install.packages("missForest"); library(missForest) #imputation with random forest
if(!("data.table" %in% installed.packages()[,"Package"])) install.packages("data.table"); library(data.table) #data tables
if(!("synthpop" %in% installed.packages()[,"Package"])) install.packages("synthpop"); library(synthpop) #synthetic population
if(!("biglm" %in% installed.packages()[,"Package"])) install.packages("biglm"); library(biglm) #synthetic population
if(!("rowr" %in% installed.packages()[,"Package"])) install.packages("rowr"); library(rowr) #cbind with fill

#NHANES sampling weight
multiplier = 1000 #each record represent multiplier number of patients

cat("\014") #clear console

# Loading Data ------------------------------------------------------------

#Importing demographic information
dem9 = read.xport("DEMO_F.XPT")
dem11 = read.xport("DEMO_G.XPT")
dem13 = read.xport("DEMO_H.XPT")
dem15 = read.xport("DEMO_I.XPT")

#Importing blood pressure readings
bp9 = read.xport("BPX_F.XPT")
bp11 = read.xport("BPX_G.XPT")
bp13 = read.xport("BPX_H.XPT")
bp15 = read.xport("BPX_I.XPT")

#Importing TC information
tc9 = read.xport("TCHOL_F.XPT")
tc11 = read.xport("TCHOL_G.XPT")
tc13 = read.xport("TCHOL_H.XPT")
tc15 = read.xport("TCHOL_I.XPT")

#Importing HDL information
hdl9 = read.xport("HDL_F.XPT")
hdl11 = read.xport("HDL_G.XPT")
hdl13 = read.xport("HDL_H.XPT")
hdl15 = read.xport("HDL_I.XPT")

#Importing LDL information
ldl9 = read.xport("TRIGLY_F.XPT")
ldl11 = read.xport("TRIGLY_G.XPT")
ldl13 = read.xport("TRIGLY_H.XPT")
ldl15 = read.xport("TRIGLY_I.XPT")

#Importing diabetes information
diab9 = read.xport("DIQ_F.XPT")
diab11 = read.xport("DIQ_G.XPT")
diab13 = read.xport("DIQ_H.XPT")
diab15 = read.xport("DIQ_I.XPT")

#Importing smoking information (SMQ680 - Used tobacco/nicotine last 5 days? for 2009-2012 and SMQ681 - Smoked tobacco last 5 days? for 2013-2016)
smoke9 = read.xport("SMQRTU_F.XPT")
smoke11 = read.xport("SMQRTU_G.XPT")
smoke13 = read.xport("SMQRTU_H.XPT")
smoke15 = read.xport("SMQRTU_I.XPT")

#Importing CVD information
cvd9 = read.xport("MCQ_F.XPT")
cvd11 = read.xport("MCQ_G.XPT")
cvd13 = read.xport("MCQ_H.XPT")
cvd15 = read.xport("MCQ_I.XPT")

#Importing drug information
drug9 = read.xport("RXQ_RX_F.XPT"); cols = sapply(drug9,is.factor); drug9[,cols] = lapply(drug9[,cols],as.character)
drug11 = read.xport("RXQ_RX_G.XPT"); cols = sapply(drug11,is.factor); drug11[,cols] = lapply(drug11[,cols],as.character)
drug13 = read.xport("RXQ_RX_H.XPT"); cols = sapply(drug13,is.factor); drug13[,cols] = lapply(drug13[,cols],as.character)
drug15 = read.xport("RXQ_RX_I.XPT"); cols = sapply(drug15,is.factor); drug15[,cols] = lapply(drug15[,cols],as.character)
drugs = toupper(c("Thiazide Diuretics","Beta-blockers","ACE inhibitors","Angiotensin II receptor antagonists","Calcium channel blockers"))

#Generating data frame of BP drug classes and generic names
bpdrugs = read.csv("List_of_Blood_Pressure_Drugs.csv")
bpdrugs = bpdrugs[which(!duplicated(bpdrugs$Generic_Name)),match(c("Class","Generic_Name"),names(bpdrugs))]
bpdrugs$Class = toupper(bpdrugs$Class); bpdrugs$Generic_Name = toupper(bpdrugs$Generic_Name)
cd = paste(bpdrugs$Generic_Name,collapse="|")

#Adding drug class to NHANES list of drugs
scodes = read.csv("Drug_Codes_and_Names_I.csv")
tmp = scodes[grep(cd,scodes$Drug_Name,ignore.case = T),]
tmp = cbind(data.frame(str_split_fixed(tmp$Drug_Name,pattern = "; ",n = 3)),tmp[,-match("Drug_Name",names(tmp))])
colnames(tmp) = c("Drug_Name1","Drug_Name2","Drug_Name3","Drug_Code")
tmp = merge(tmp,bpdrugs,by.x = "Drug_Name1",by.y = "Generic_Name",all.x = T)
drugs_classes = tmp[which(!is.na(tmp$Class)),]; rm(tmp)
cols = sapply(drugs_classes,is.factor); drugs_classes[,cols] = lapply(drugs_classes[,cols],as.character)

# #Removing irrelevant medications by type (not necessary for current dataset)
# tmp = tmp[-grep("OPHTHALMIC",tmp$Drug_Name1,ignore.case = T),] #removing glaucoma medications from first drug name
# tmp = tmp[-grep("OPHTHALMIC",tmp$Drug_Name2,ignore.case = T),] #removing glaucoma medications from second drug name
# tmp = tmp[-grep("OPHTHALMIC",tmp$Drug_Name3,ignore.case = T),] #removing glaucoma medications from second drug name
# 
# ##Subsetting matched and unmatched drugs (not necessary for current dataset - all drugs are matched)
# tmp1 = tmp[which(!is.na(tmp$Class)),]
# tmp2 = tmp[which(is.na(tmp$Class)),]
# 
# ##Matching by second drug (if necessary; n=3 in str_split_fixed)
# tmp2 = merge(tmp2[,-match("Class",names(tmp))],bpdrugs,by.x = "Drug_Name2",by.y = "Generic_Name",all.x = T)
# 
# ##Subsetting matched and unmatched drugs (not necessary for current dataset)
# tmp3 = tmp2[which(!is.na(tmp2$Class)),]
# tmp4 = tmp2[which(is.na(tmp2$Class)),]
# 
# ##Matching by third drug (if necessary; n=3 in str_split_fixed)
# tmp4 = merge(tmp4,bpdrugs,by.x = "Drug_Name3",by.y = "Generic_Name",all.x = T)
# 
# ##Merging complete drugs with classes
# drugs_classes = rbind(tmp1,tmp2); rm(tmp,tmp1,tmp2,tmp3,tmp4)

#Creating data frame with medication counts and medication combinations for each data file
##2009-2010
seqn = drug9$SEQN[which(is.element(drug9$RXDDRGID,drugs_classes$Drug_Code))]
drg = drug9$RXDDRGID[which(is.element(drug9$RXDDRGID,drugs_classes$Drug_Code))]
drg_cls = drugs_classes$Class[match(drg,drugs_classes$Drug_Code)]
temp = data.frame(seqn,drg,drg_cls); rm(seqn,drg,drg_cls)

drg_cls = num_bp = c(); j=1
for(i in unique(temp$seqn)){
  tmp = as.character(temp[which(temp$seqn==i),match("drg_cls",names(temp))])
  
  num_bp[j] = length(tmp)
  drg_cls[j] = paste0(ifelse(is.element(tmp,drugs),tmp,"OTHER"),collapse = "+")
  rm(tmp)
  
  j=j+1
}

new_drug9 = data.frame(unique(temp$seqn),num_bp,drg_cls); rm(temp)
colnames(new_drug9) = c("SEQN","BPCOUNT","BPCOMB")

##2011-2012
seqn = drug11$SEQN[which(is.element(drug11$RXDDRGID,drugs_classes$Drug_Code))]
drg = drug11$RXDDRGID[which(is.element(drug11$RXDDRGID,drugs_classes$Drug_Code))]
drg_cls = drugs_classes$Class[match(drg,drugs_classes$Drug_Code)]
temp = data.frame(seqn,drg,drg_cls); rm(seqn,drg,drg_cls)

drg_cls = num_bp = c(); j=1
for(i in unique(temp$seqn)){
  tmp = as.character(temp[which(temp$seqn==i),match("drg_cls",names(temp))])
  
  num_bp[j] = length(tmp)
  drg_cls[j] = paste0(ifelse(is.element(tmp,drugs),tmp,"OTHER"),collapse = "+")
  rm(tmp)
  
  j=j+1
}

new_drug11 = data.frame(unique(temp$seqn),num_bp,drg_cls); rm(temp)
colnames(new_drug11) = c("SEQN","BPCOUNT","BPCOMB")

##2013-2014
seqn = drug13$SEQN[which(is.element(drug13$RXDDRGID,drugs_classes$Drug_Code))]
drg = drug13$RXDDRGID[which(is.element(drug13$RXDDRGID,drugs_classes$Drug_Code))]
drg_cls = drugs_classes$Class[match(drg,drugs_classes$Drug_Code)]
temp = data.frame(seqn,drg,drg_cls); rm(seqn,drg,drg_cls)

drg_cls = num_bp = c(); j=1
for(i in unique(temp$seqn)){
  tmp = as.character(temp[which(temp$seqn==i),match("drg_cls",names(temp))])
  
  num_bp[j] = length(tmp)
  drg_cls[j] = paste0(ifelse(is.element(tmp,drugs),tmp,"OTHER"),collapse = "+")
  rm(tmp)
  
  j=j+1
}

new_drug13 = data.frame(unique(temp$seqn),num_bp,drg_cls); rm(temp)
colnames(new_drug13) = c("SEQN","BPCOUNT","BPCOMB")


##2015-2016
seqn = drug15$SEQN[which(is.element(drug15$RXDDRGID,drugs_classes$Drug_Code))]
drg = drug15$RXDDRGID[which(is.element(drug15$RXDDRGID,drugs_classes$Drug_Code))]
drg_cls = drugs_classes$Class[match(drg,drugs_classes$Drug_Code)]
temp = data.frame(seqn,drg,drg_cls); rm(seqn,drg,drg_cls)

drg_cls = num_bp = c(); j=1
for(i in unique(temp$seqn)){
  tmp = as.character(temp[which(temp$seqn==i),match("drg_cls",names(temp))])
  
  num_bp[j] = length(tmp)
  drg_cls[j] = paste0(ifelse(is.element(tmp,drugs),tmp,"OTHER"),collapse = "+")
  rm(tmp)
  
  j=j+1
}

new_drug15 = data.frame(unique(temp$seqn),num_bp,drg_cls); rm(temp)
colnames(new_drug15) = c("SEQN","BPCOUNT","BPCOMB")

#Joining datasets
dfs = list(dem9,bp9,tc9,hdl9,ldl9,diab9,smoke9,cvd9,new_drug9)
data9 = Reduce(function(...) merge(..., all.x=T,by = "SEQN"), dfs)

dfs = list(dem11,bp11,tc11,hdl11,ldl11,diab11,smoke11,cvd11,new_drug11)
data11 = Reduce(function(...) merge(..., all.x=T,by = "SEQN"), dfs)

dfs = list(dem13,bp13,tc13,hdl13,ldl13,diab13,smoke13,cvd13,new_drug13)
data13 = Reduce(function(...) merge(..., all.x=T,by = "SEQN"), dfs)

dfs = list(dem15,bp15,tc15,hdl15,ldl15,diab15,smoke15,cvd15,new_drug15)
data15 = Reduce(function(...) merge(..., all.x=T,by = "SEQN"), dfs)

rm(list = ls()[!(ls() %in% c("home_dir","main_dir","multiplier","data9","data11","data13","data15"))]); gc()

#Calculating 8-year weights
data9$WTMEC8YR = data9$WTMEC2YR*1/4 #WTMEC2YR was used instead of WTINT2YR because some variables were taken in the MEC
data11$WTMEC8YR = data11$WTMEC2YR*1/4
data13$WTMEC8YR = data13$WTMEC2YR*1/4
data15$WTMEC8YR = data15$WTMEC2YR*1/4

#Removing Unnesary Information
#change SMQ680 or SMQ681 for SMQ040 if using SMQ files intead of SMQRTU files for smoking status
data9 = data.frame("YEARS" = "2009-2010",data9[,c("SEQN","WTMEC8YR","RIDAGEYR","RIAGENDR","RIDRETH1","BPXSY1","BPXDI1","LBXTC","LBDHDD","LBDLDL","SMQ680","DIQ010","MCQ160F","MCQ160E","BPCOUNT","BPCOMB")])
data11 = data.frame("YEARS" = "2011-2012",data11[,c("SEQN","WTMEC8YR","RIDAGEYR","RIAGENDR","RIDRETH1","BPXSY1","BPXDI1","LBXTC","LBDHDD","LBDLDL","SMQ680","DIQ010","MCQ160F","MCQ160E","BPCOUNT","BPCOMB")])
data13 = data.frame("YEARS" = "2013-2014",data13[,c("SEQN","WTMEC8YR","RIDAGEYR","RIAGENDR","RIDRETH1","BPXSY1","BPXDI1","LBXTC","LBDHDD","LBDLDL","SMQ681","DIQ010","MCQ160F","MCQ160E","BPCOUNT","BPCOMB")])
data15 = data.frame("YEARS" = "2015-2016",data15[,c("SEQN","WTMEC8YR","RIDAGEYR","RIAGENDR","RIDRETH1","BPXSY1","BPXDI1","LBXTC","LBDHDD","LBDLDL","SMQ681","DIQ010","MCQ160F","MCQ160E","BPCOUNT","BPCOMB")])
colnames(data13) = colnames(data15) = colnames(data11)

#Combining datasets
ndata = rbind(data9,data11,data13,data15)
rm(list = ls()[!(ls() %in% c("home_dir","main_dir","multiplier","ndata"))]); gc()

save(ndata,file = "Raw_Continuous_NHANES_Dataset.RData")

# Modifying Dataset ------------------------------------------------------

#Loading data
load("Raw_Continuous_NHANES_Dataset.RData")

#Renaming columns
cnames = read.csv("Continuous_NHANES_Variables.csv")
colnames(ndata) = as.character(cnames$Meaning[match(names(ndata),as.character(cnames$Variable))]) #renaming columns

#Counting records
nrow(ndata)
sum(ndata$WEIGHT)/1e06

#Restricting Analysis for ages 40 to 75
length(ndata$WEIGHT[ndata$AGE<40 | ndata$AGE>75]) #age exclusions
sum(ndata$WEIGHT[ndata$AGE<40 | ndata$AGE>75])/1e06 #age exclusions
ndata = ndata[ndata$AGE>=40 & ndata$AGE<=75,]

ndata$SEX = factor(ndata$SEX) #Converting gender into categorical variable
levels(ndata$SEX)=list("Male"=1,"Female"=2)

ndata$RACE = factor(ndata$RACE) #Converting race into categorical variable
levels(ndata$RACE)=list("White"=3,"Black"=4,"Hispanic"=1,"Hispanic"=2,"Other"=5)

ndata$SMOKER = factor(ndata$SMOKER) #Converting smoking into categorical variable
levels(ndata$SMOKER)=list("N"=2,"Y"=1,"U"=7,"U"=9)#;relevel(ndata$SMOKER,ref = "N")

ndata$DIABETIC = factor(ndata$DIABETIC) #Converting diabetic into categorical variable
levels(ndata$DIABETIC)=list("N"=2,"N"=3,"Y"=1,"U"=7,"U"=9)#;relevel(ndata$DIABETIC,ref = "N")

ndata$STROKE = factor(ndata$STROKE) #Converting stroke into categorical variable
levels(ndata$STROKE)=list("N"=2,"Y"=1,"U"=7,"U"=9)#;relevel(ndata$STROKE,ref = "N")

ndata$CHD = factor(ndata$CHD) #Converting CHD into categorical variable
levels(ndata$CHD)=list("N"=2,"Y"=1,"U"=7,"U"=9)#;relevel(ndata$CHD,ref = "N")

#Removing Unknowns
ndata[ndata == "U"] = NA

#Removing 0's from DBP
ndata$DBP = ifelse(ndata$DBP==0,NA,ndata$DBP)

#Removing DBPs with numbers too low
ndata$DBP = ifelse(ndata$DBP<10,NA,ndata$DBP)

#Excluding patients with history of CHD or stroke and removing variables
length(ndata$WEIGHT[which(ndata$CHD=="Y"|ndata$STROKE=="Y")]) #history of CVD exclusions
sum(ndata$WEIGHT[which(ndata$CHD=="Y"|ndata$STROKE=="Y")]) #history of CVD exclusions
ndata = ndata[-which(ndata$CHD=="Y"|ndata$STROKE=="Y"),]
cols = match(c("CHD","STROKE"),names(ndata))
ndata = ndata[,-cols]

#Excluding Hipanic and Other race patients (not in ASCVD risk)
length(ndata$WEIGHT[which(ndata$RACE=="Hispanic"|ndata$RACE=="Other")]) #race exclusions
sum(ndata$WEIGHT[which(ndata$RACE=="Hispanic"|ndata$RACE=="Other")]) #race exclusions
ndata = ndata[-which(ndata$RACE=="Hispanic"|ndata$RACE=="Other"),]

#Changing NAs to "No TREATMENT" in drug variables
ndata$BPCOUNT[which(is.na(ndata$BPCOUNT))] = 0
levels(ndata$BPCOMB) = c(levels(ndata$BPCOMB),"NO TREATMENT")
ndata$BPCOMB[which(ndata$BPCOUNT==0)] = "NO TREATMENT"

#Counting dataset after exclusions
ndata = droplevels(ndata)
length(ndata$WEIGHT)
sum(ndata$WEIGHT)/1e06
save(ndata,file = "Continuous_NHANES_Dataset.RData")

# Performing Multiple Imputation ------------------------------------------

#Loading preprocessed dataset
load("Continuous_NHANES_Dataset.RData")

#Removing drug combination for imputation (it has more than 53 levels)
BPCOMB = ndata$BPCOMB
ndata = ndata[,-match("BPCOMB",names(ndata))]

#Parallel computing parameters
cl = makeCluster(ncol(ndata))
registerDoParallel(cl)

#Imputing data using random forest
set.seed(123)
impdata = missForest(ndata, parallelize = "forest")
impdata = impdata$ximp

stopCluster(cl)

#Adding age squared column and reorganizing columns
impdata$AGE2 = impdata$AGE^2
cols = match(c("YEARS","SEQN","WEIGHT","AGE","AGE2","SEX","RACE","SMOKER","DIABETIC","SBP","DBP","TC","HDL","LDL","BPCOUNT"),names(impdata))
impdata = impdata[,cols]

#Adding drug combination back
impdata = cbind(impdata,BPCOMB)

save(impdata,file = "Continuous NHANES Imputed Dataset.RData")

# Generating Dataset Representative of US Population ----------------------

#Loading preprocessed dataset
load("Continuous NHANES Imputed Dataset.RData")

#Coverting to data table
setDT(impdata)

#Generating new data
usdata = impdata[rep(seq(.N), round(impdata$WEIGHT/multiplier))]

#Counting records
length(usdata$WEIGHT) #Total records
length(usdata$WEIGHT)*multiplier/1e06 #Population size

#Removing unnecessary varibles
usdata[, c("WEIGHT"):=NULL]

#Saving information as CSV
fwrite(usdata,"Continuous NHANES US Population.csv")

# Generating Synthetic Dataset --------------------------------------------

#Loading preprocessed dataset
setwd(home_dir)
usdata = fread("Continuous NHANES US Population.csv")

set.seed(456)
synthetic = syn(usdata,k = nrow(usdata), method = c("","","cart",~I(AGE^2),"cart","cart","cart","cart",
                                                    "cart","cart","cart","cart","cart","",""),
                smoothing = list(AGE="density",SBP="density",DBP="density",TC="density",HDL="density",LDL="density")) #synthesizing dataset
syndata = synthetic$syn #Extracting synthetic dataset

#Removing treated patients (revision 1/3/2020)
setDT(syndata)
syndata = syndata[BPCOMB=="NO TREATMENT",]
syndata[,c("BPCOUNT","BPCOMB"):=NULL]

fwrite(syndata,"Continuous NHANES Synthetic Dataset.csv")

# Linear Models for Continuous Variables ----------------------------------

#Loading preprocessed dataset
setwd(home_dir)
syndata = fread("Continuous NHANES Synthetic Dataset.csv")

#Converting categorical variable into factors
fcols = c("YEARS","SEX","RACE","SMOKER","DIABETIC") #,"BPCOMB"
syndata[,(fcols):=lapply(.SD, as.factor),.SDcols=fcols]

#Setting no treatment as reference level in drug treatment
# syndata$BPCOMB = relevel(syndata$BPCOMB,ref = "NO TREATMENT")
syndata$SEX = relevel(syndata$SEX,ref = "Male")
syndata$RACE = relevel(syndata$RACE,ref = "White")

#Counting final dataset
nrow(syndata)
nrow(syndata)*multiplier/1e06

#Defining columns of predictors
cols = match(c("AGE","AGE2","SEX","RACE","SMOKER","DIABETIC"),names(syndata)) #,"BPCOUNT","BPCOMB"

#Linear regression for SBP
fm = as.formula(paste("SBP~",paste(colnames(syndata)[cols],collapse = "+")))
full = lm(fm,data=syndata) #Full model
svs = step(full,direction="both",trace = 0) #Stepwise variable selection
sbp = biglm(formula(svs),data=syndata) #memory efficient lm for saving

print(paste("sbp done ",Sys.time(),sep = ""))

#Linear regression for DBP
fm = as.formula(paste("DBP~",paste(colnames(syndata)[cols],collapse = "+")))
full = lm(fm,data=syndata) #Full model
svs = step(full,direction="both",trace = 0) #Stepwise variable selection
dbp = biglm(formula(svs),data=syndata) #memory efficient lm for saving

print(paste("dbp done ",Sys.time(),sep = ""))

#Linear regression for TC
fm = as.formula(paste("TC~",paste(colnames(syndata)[cols],collapse = "+")))
full = lm(fm,data=syndata) #Full model
svs = step(full,direction="both",trace = 0) #Stepwise variable selection
tc = biglm(formula(svs),data=syndata) #memory efficient lm for saving

print(paste("tc done ",Sys.time(),sep = ""))

#Linear regression for HDL
fm = as.formula(paste("HDL~",paste(colnames(syndata)[cols],collapse = "+")))
full = lm(fm,data=syndata) #Full model
svs = step(full,direction="both",trace = 0) #Stepwise variable selection
hdl = biglm(formula(svs),data=syndata) #memory efficient lm for saving

print(paste("hdl done ",Sys.time(),sep = ""))

#Linear regression for LDL
fm = as.formula(paste("LDL~",paste(colnames(syndata)[cols],collapse = "+")))
full = lm(fm,data=syndata) #Full model
svs = step(full,direction="both",trace = 0) #Stepwise variable selection
ldl = biglm(formula(svs),data=syndata) #memory efficient lm for saving

print(paste("ldl done ",Sys.time(),sep = ""))

#Deleting unnecesary files and cleaning memory
rm(full,svs);gc()

save(sbp,dbp,tc,hdl,ldl,file = "Linear regression models.RData")

#Table of regression coeficients
rc = cbind.fill(sbp$names,coef(sbp),coef(dbp),coef(hdl),coef(tc),fill=NA)
colnames(rc) = c("sbp","dbp","hdl","tc")
write.csv(rc,"Regression Coefficients.csv")

# Forecasting Continuous Variables ----------------------------------------

print(paste("Performing forecast ",Sys.time(),sep = ""))

#Loading data
setwd(home_dir)
# syndata = fread("Continuous NHANES Synthetic Dataset.csv") #expanded syntethic dataset
load("Continuous NHANES Imputed Dataset.RData") # imputed dataset
syndata = impdata # using all ages without expanding the dataset according to sampling weights
# syndata = impdata[impdata$AGE>=50 & impdata$AGE<55,] # using ages 50 to 54 for main analysis in Python
# syndata = impdata[impdata$AGE>=70 & impdata$AGE<75,] # using ages 70 to 74 for sensitivity analysis in Python
setDT(syndata)
load("Linear regression models.RData")

#Converting categorical variable into factors
fcols = c("SEX","RACE","SMOKER","DIABETIC") #,"BPCOMB"
syndata[,(fcols):=lapply(.SD, as.factor),.SDcols=fcols]

#Setting no treatment as reference level in drug treatment
# syndata$BPCOMB = relevel(syndata$BPCOMB,ref = "NO TREATMENT")
syndata$SEX = relevel(syndata$SEX,ref = "Male")
syndata$RACE = relevel(syndata$RACE,ref = "White")

print(paste("Loading done ",Sys.time(),sep = ""))

#Expanding dataset for ten year forecast
patientdata = syndata[rep(seq(.N), each=10)]

print(paste("Expansion done ",Sys.time(),sep = ""))

#Saving age from syndata
age = syndata[,AGE]

#Caculating intercept adjustements
intercept_sbp = rep(c(syndata[,SBP]-predict(sbp,newdata = syndata)),each=10)
intercept_dbp = rep(c(syndata[,DBP]-predict(dbp,newdata = syndata)),each=10)
intercept_tc = rep(c(syndata[,TC]-predict(tc,newdata = syndata)),each=10)
intercept_hdl = rep(c(syndata[,HDL]-predict(hdl,newdata = syndata)),each=10)
intercept_ldl = rep(c(syndata[,LDL]-predict(ldl,newdata = syndata)),each=10)

#Removing syndata from memory
rm(syndata);gc()

#Filling age
patientdata[,AGE := (rep(age,each = 10)+0:9)]

#Calculating squared age
patientdata[,AGE2 := AGE^2]

print(paste("Age done ",Sys.time(),sep = ""))

#Linear regression for SBP
patientdata[,SBP := predict(sbp,newdata = patientdata)+(intercept_sbp)]
print(paste("sbp done ",Sys.time(),sep = ""))

#Linear regression for DBP
patientdata[,DBP := predict(dbp,newdata = patientdata)+(intercept_dbp)]
print(paste("dbp done ",Sys.time(),sep = ""))

#Linear regression for TC
patientdata[,TC := predict(tc,newdata = patientdata)+(intercept_tc)]
print(paste("tc done ",Sys.time(),sep = ""))

#Linear regression for HDL
patientdata[,HDL := predict(hdl,newdata = patientdata)+(intercept_hdl)]
print(paste("hdl done ",Sys.time(),sep = ""))

#Linear regression for LDL
patientdata[,LDL := predict(ldl,newdata = patientdata)+(intercept_ldl)]
print(paste("ldl done ",Sys.time(),sep = ""))

#Changing SEQN number for ordered ID
patientdata[,SEQN := rep(0:((nrow(patientdata)-1)/10), each=10)]

nrow(patientdata)/10 #Total participants
nrow(patientdata)/10*multiplier #Total population

#Removing unnecessary varibles
patientdata[, c("YEARS","AGE2","BPCOUNT","BPCOMB"):=NULL] #,"BPCOUNT","BPCOMB"

#Formatting data for Computations
patientdata[, SEX := ifelse(SEX=="Male",1,0)]
patientdata[, RACE := ifelse(RACE=="White",1,0)]
patientdata[, SMOKER := ifelse(SMOKER=="Y",1,0)]
patientdata[, DIABETIC := ifelse(DIABETIC=="Y",1,0)]
colnames(patientdata) = c("id","wt","age","sex","race","smk","diab","sbp","dbp","tc","hdl","ldl") #Column names for Continuous NHANES

# save(patientdata,file = "Continuous NHANES Synthetic Forecasted Dataset.RData")
# fwrite(patientdata,"Continuous NHANES Synthetic Forecasted Dataset.csv")
# fwrite(patientdata,"Continuous NHANES 50-54 Dataset.csv")
# fwrite(patientdata,"Continuous NHANES 70-74 Dataset.csv")
fwrite(patientdata,"Continuous NHANES 40-75 Dataset.csv")


# Patient Profiles --------------------------------------------------------

#Loading data
setwd(home_dir)
profiledata = fread("Patient Profile Data.csv")
load("Linear regression models.RData")

#Saving age from profiledata
age = profiledata[,AGE]

#Caculating intercept adjustements
intercept_sbp = rep(c(profiledata[,SBP]-predict(sbp,newdata = profiledata)),each=10)
intercept_dbp = rep(c(profiledata[,DBP]-predict(dbp,newdata = profiledata)),each=10)
intercept_tc = rep(c(profiledata[,TC]-predict(tc,newdata = profiledata)),each=10)
intercept_hdl = rep(c(profiledata[,HDL]-predict(hdl,newdata = profiledata)),each=10)

#Expanding profiles for ten year forecast
profiledata = profiledata[rep(seq(.N), each=10)]

#Filling age
profiledata[,AGE := (rep(age,each = 10)+0:9)]

#Linear regression for SBP
profiledata[,SBP := predict(sbp,newdata = profiledata)+(intercept_sbp)]

#Linear regression for DBP
profiledata[,DBP := predict(dbp,newdata = profiledata)+(intercept_dbp)]

#Linear regression for TC
profiledata[,TC := predict(tc,newdata = profiledata)+(intercept_tc)]

#Linear regression for HDL
profiledata[,HDL := predict(hdl,newdata = profiledata)+(intercept_hdl)]

#Removing unnecesary columns
profiledata[, c("AGE2"):=NULL]

#Formatting data for Computations
profiledata[, SEX := ifelse(SEX=="Male",1,0)]
profiledata[, RACE := ifelse(RACE=="White",1,0)]
profiledata[, SMOKER := ifelse(SMOKER=="Y",1,0)]
profiledata[, DIABETIC := ifelse(DIABETIC=="Y",1,0)]
colnames(profiledata) = c("id","age","sex","race","smk","diab","sbp","dbp","tc","hdl")

# Saving forecasted patient profiles
fwrite(profiledata,"Complete Forecasted Patient Profiles.csv")
