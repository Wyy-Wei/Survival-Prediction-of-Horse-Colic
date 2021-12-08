
rm(list=ls())
horse.colic <- read.table("D:/chrome_download/STA5703/finalproject/horsecolic/horse-colic.data",
                          quote="\"", comment.char="", na.strings="?")
horse.colic.test <- read.table("D:/chrome_download/STA5703/finalproject/horsecolic/horse-colic.test",
                               quote="\"", comment.char="", na.strings="?")
horse <- rbind(horse.colic,horse.colic.test)

# change column names to readable names
names <- c("surgery","age","hospital_num","rectal_temp","pulse","respi_rate",
           "t_extremities","peri_pulse","color","refill_time","pain","peristalsis",
           "abd_distend","gas_tube","gas_reflux","reflux_ph","feces","abdomen",
           "cell_vol","total_protein","abd_appea","abd_protein","outcome",
           "surgical_les","lesion1","lesion2","lesion3","cp")
colnames(horse) <- names

# delete the data without response variable "outcome"
horse <- horse[-which(is.na(horse$outcome)),]

library(ggplot2)
missing <- data.frame(m=colMeans(is.na(horse)),n=names)
missing <- missing[as.numeric(which(missing$m>0.2)),]
ggplot(data=missing, aes(x=reorder(n, -m), y=m)) + 
  geom_bar(stat = "identity", fill="steelblue") + xlab("variables") + 
  ylab("proportion of missing value")

# delete the columns with more than 50% missing values
# that is, reflux_ph,abd_appea and abd_protein
horse <- horse[,-c(which(colMeans(is.na(horse))>=0.5))]

# delete hospital number and cp data
horse <- subset(horse, select = (-c(hospital_num, cp)))

# convert outcome to be either 1 or 0
horse$outcome[horse$outcome!=1] <- 0

# replace missing values with 0
# because for logistic regression
horse[is.na(horse)] <- 0

# make sure the order of ordinal variables
horse$t_extremities[horse$t_extremities==1] <- 99
horse$t_extremities[horse$t_extremities==2] <- 1
horse$t_extremities[horse$t_extremities==99] <- 2

horse$peri_pulse[horse$peri_pulse==1] <- 99
horse$peri_pulse[horse$peri_pulse==2] <- 1
horse$peri_pulse[horse$peri_pulse==99] <- 2

horse$feces[horse$feces==1] <- 99
horse$feces[horse$feces==2] <- 1
horse$feces[horse$feces==99] <- 2

train <- horse[1:299,]
test <- horse[300:366,]

# Convert factors to dummy variables
horse$surgery[horse$surgery==1] <- "yes"
horse$surgery[horse$surgery==2] <- "no"
horse$surgery <- as.factor(horse$surgery)

horse$age[horse$age==1] <- "adult"
horse$age[horse$age==9] <- "young" #typo
horse$age <- as.factor(horse$age)

horse$color[horse$color==1] <- "normal_pink"
horse$color[horse$color==2] <- "bright_pink"
horse$color[horse$color==3] <- "pale_pink"
horse$color[horse$color==4] <- "pale_cyanotic"
horse$color[horse$color==5] <- "bright_red_or_injected"
horse$color[horse$color==6] <- "dark_cyanotic"
horse$color <- as.factor(horse$color)

horse$refill_time[horse$refill_time==1] <- "<3s"
horse$refill_time[horse$refill_time==2] <- ">=3s" 
horse$refill_time <- as.factor(horse$refill_time)

horse$pain[horse$pain==1] <- "alert_no_pain"
horse$pain[horse$pain==2] <- "depressed"
horse$pain[horse$pain==3] <- "intermittent_mild_pain"
horse$pain[horse$pain==4] <- "intermittent_severe_pain"
horse$pain[horse$pain==5] <- "continuous_severe_pain"
horse$pain <- as.factor(horse$pain)

horse$gas_reflux[horse$gas_reflux==1] <- "none"
horse$gas_reflux[horse$gas_reflux==2] <- ">1L"
horse$gas_reflux[horse$gas_reflux==3] <- "<1L"
horse$gas_reflux <- as.factor(horse$gas_reflux)

horse$abdomen[horse$abdomen==1] <- "normal"
horse$abdomen[horse$abdomen==2] <- "other"
horse$abdomen[horse$abdomen==3] <- "firm_feces_in_the_large_intestine"
horse$abdomen[horse$abdomen==4] <- "distended_small_intestine"
horse$abdomen[horse$abdomen==5] <- "distended_large_intestine"
horse$abdomen <- as.factor(horse$abdomen)

horse$surgical_les[horse$surgical_les==1] <- "yes"
horse$surgical_les[horse$surgical_les==2] <- "no"
horse$surgical_les <- as.factor(horse$surgical_les)

surgery <- model.matrix(~0+horse[,"surgery"])
age <- model.matrix(~0+horse[,"age"])
color <- model.matrix(~0+horse[,"color"])
refill_time <- model.matrix(~0+horse[,"refill_time"])
pain <- model.matrix(~0+horse[,"pain"])
gas_reflux <- model.matrix(~0+horse[,"gas_reflux"])
abdomen <- model.matrix(~0+horse[,"abdomen"])
surgical_les <- model.matrix(~0+horse[,"surgical_les"])

horse <- subset(horse, select = -c(surgery,age,color,refill_time,
                                   pain,gas_reflux,abdomen,surgical_les))
horse <- cbind(horse,surgery,age,color[,-1],refill_time[,-c(2,3)],
               pain[,-1],gas_reflux[,-3],abdomen[,-1],surgical_les)
# now there are 43 columns
dummynames <- c("surgery_no","surgery_yes","adult","young","bright_pink",
              "bright_red_or_injected","dark_cyanotic","normal_pink",
              "pale_cyanotic","pale_pink","refill_time_less_than_3s",
              "refill_time_greater_than_3s","alert_no_pain",
              "continuous_severe_pain","depressed_pain",
              "intermittent_mild_pain","intermittent_severe_pain",
              "gas_reflux_less_than_1L","gas_reflux_greater_than_1L",
              "gas_reflux_none","abdomen_distended_large_intestine",
              "abdomen_distended_small_intestine","abdoman_firm_feces_large",
              "abdomen_normal","abdomen_other","surgical_les_no","surgical_les_yes")
colnames(horse) <- c(colnames(horse[,1:15]),dummynames)

# Split the data into training and testing datasets
# Convert dataframes to Dmatrixs
library(xgboost)
horse_train <- as.matrix(horse[1:299,-12])
horse_trainlabel <- as.matrix(horse[1:299,12])
horse_dtrain <- xgb.DMatrix(data = horse_train, label = horse_trainlabel)
horse_test <- as.matrix(horse[300:366,-12])
horse_testlabel <- as.matrix(horse[300:366,12])
horse_dtest <- xgb.DMatrix(data = horse_test, label = horse_testlabel)

### XGBoost
# train the model
xgb_model <- xgboost(data = horse_dtrain, nrounds = 2,
                     objective = "binary:logistic")
## [1]	train-error:0.130435 
## [2]	train-error:0.113712 

# generate predictions for our held out testing data 
xgb_pred <- predict(xgb_model, horse_dtest)

# get and print the classification error
error <- mean(as.numeric(xgb_pred > 0.5) != horse_testlabel)
print(paste("Test-error =", error))
## [1] "Test-error = 0.313432835820896"

# tuning parameters
params <- list(booster = "gbtree", objective = "binary:logistic",
               eta=0.3, gamma=0, max_depth=6, min_child_weight=1, 
               subsample=1, colsample_bytree=1)
set.seed(123)
xgbcv <- xgb.cv( params = params, data = horse_dtrain, nrounds = 100, 
                 nfold = 5, showsd = T, stratified = T, maximize = F,
                 print.every.n = 10, early.stop.round = 20)
## Best iteration: [10]	train-error:0.011737+0.005645	test-error:0.243368+0.069367
xgb1 <- xgb.train (params = params, data = horse_dtrain, nrounds = 10, 
                   watchlist = list(val=horse_dtest,train=horse_dtrain), 
                   print.every.n = 10, early.stop.round = 10, 
                   maximize = F , eval_metric = "error")
xgbpred1 <- predict (xgb1,horse_dtest)
error1 <- mean(as.numeric(xgb_pred > 0.5) != horse_testlabel)
error1 #[1] 0.3134328
mat <- xgb.importance (feature_names = colnames(horse[,-12]),model = xgb1)
xgb.plot.importance (importance_matrix = mat[1:15]) 
# xgb.plot.multi.trees(feature_names = colnames(horse_train[,-12]), model = xgb1)

library(mlr)
#create tasks
horse$outcome <- as.factor(horse$outcome)
traintask <- makeClassifTask (data = as.data.frame(horse[1:299,]),target = "outcome")
testtask <- makeClassifTask (data = as.data.frame(horse[300:366,]),target = "outcome")

#do one hot encoding 
traintask <- createDummyFeatures (obj = traintask,target = "outcome") 
testtask <- createDummyFeatures (obj = testtask,target = "outcome")

#create learner
lrn <- makeLearner("classif.xgboost",predict.type = "response")
lrn$par.vals <- list( objective="binary:logistic", eval_metric="error", nrounds=100L, eta=0.1)

#set parameter space
params <- makeParamSet( makeDiscreteParam("booster",values = c("gbtree")), 
                        makeIntegerParam("max_depth",lower = 3L,upper = 10L), 
                        makeNumericParam("min_child_weight",lower = 1L,upper = 10L), 
                        makeNumericParam("subsample",lower = 0.5,upper = 1), 
                        makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))

#set resampling strategy
rdesc <- makeResampleDesc("CV",stratify = T,iters=5L)

#search strategy
ctrl <- makeTuneControlRandom(maxit = 10L)

#parameter tuning
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, 
                     measures = acc, par.set = params, control = ctrl, show.info = F)
mytune  # 0.7727119 

#set hyperparameters
lrn_tune <- setHyperPars(lrn,par.vals = mytune$x)

#train model
xgb_tuned <- mlr::train(learner = lrn_tune,task = traintask)
# train-error:0.073579

pred_tuned <- predict(xgb_tuned, testtask)
library(caret)
confusionMatrix(pred_tuned$data$response,pred_tuned$data$truth)

error_tuned <- mean(pred_tuned$data$response!=pred_tuned$data$truth)
print(paste("Test-error =", error_tuned))
# [1] "Test-error = 0.194029850746269"


conf <- as.data.frame(prop.table(table(pred_tuned$data$response,horse_testlabel),2))
ggplot(conf) + geom_tile(aes(x=horse_testlabel,y=Var1, fill=Freq)) + guides(fill=F)+
  scale_y_discrete(name="Actual Class") + scale_x_discrete(name="Predicted Class")+
  geom_text(aes(y=Var1,x=horse_testlabel, label=round(Freq,3)), color="black") + 
  scale_fill_distiller(palette="Blues",direction = 1)

# mytune$x
# $`booster` [1] "gbtree"
# $max_depth [1] 3
# $min_child_weight [1] 1.339608
# $subsample [1] 0.5236741
# $colsample_bytree [1] 0.5804795

set.seed(123)
xgbcv_tuned <- xgb.cv( eta=0.1,max_depth=5, min_child_weight=3.629998, 
                       subsample=0.5812887, colsample_bytree=0.6754075, 
                       data = horse_dtrain, nrounds = 100, 
                 nfold = 5, showsd = T, stratified = T, maximize = F,
                 print.every.n = 10, early.stop.round = 20)
# Best iteration: [45]	train-rmse:0.189132+0.006688	test-rmse:0.399840+0.043151
model_tuned <- xgboost(data = horse_dtrain, nrounds = 45, eta=0.1,
                       max_depth=5, min_child_weight=3.629998, 
                       subsample=0.5812887, colsample_bytree=0.6754075, 
                       objective = "binary:logistic")
pred_tuned <- predict(model_tuned, horse_dtest)

error_tuned <- mean(as.numeric(pred_tuned > 0.5) != horse_testlabel)
print(paste("Test-error =", error_tuned))
#  "Test-error = 0.208955223880597"

importance_matrix_tuned <- xgb.importance(colnames(horse[,-12]), model = model_tuned)
xgb.plot.importance(importance_matrix_tuned[1:15])
# xgb.plot.multi.trees(feature_names = colnames(horse[,-12]), model = model_tuned)

library(breakDown)
nobs <- horse_train[1L, , drop = FALSE]

explain_2 <- broken(model_tuned, new_observation = nobs, 
                    data = horse_train)
explain_2
library(ggplot2)
plot(explain_2)


library(psych)
X = scale(train[,-19],center = T,scale = T)
m = fa.parallel(X,fm="gls") 
# Parallel analysis suggests that the number of factors =  6  and the number of components =  4 
x.fa2 = fa(X,nfactors = 6,fm='mle',rotate = 'varimax')
## fm="pa" will do the principal factor solution
##fm="alpha" will do alpha factor analysis as described in Kaiser and Coffey (1965)
round(x.fa2$loadings,6)
fa.diagram(x.fa2) 

pc = principal(X,nfactors = 4,rotate='varimax')
pc$loadings
fa.diagram(pc,simple=T)
factor.plot(pc$loadings,labels = rownames(pc$loadings))

pca<-princomp(train[,-19],cor=T)
summary(pca,loadings=T)
screeplot(pca,type="l")

# Stepwise logistic regression 

library(MASS)

train <- as.data.frame(horse[1:299,])
test <- as.data.frame(horse[300:366,])
m1<-glm(outcome~.,data = train,family = binomial(link="logit"))
m0<-stepAIC(m1,trace = FALSE)
summary(m1)
summary(m0)
# Make predictions
p1<-predict(m1,train,type="response")
predicted.classes1 <- ifelse(p1 > 0.5, "1", "0")
# Prediction accuracy
observed.classes1 <- horse[1:299,]$outcome
mean(predicted.classes1 == observed.classes1)
# [1] 0.7658863

# Make predictions
p0<-predict(m0,train,type="response")
predicted.classes0 <- ifelse(p0 > 0.5, "1", "0")
# Prediction accuracy
observed.classes0 <- horse$outcome
mean(predicted.classes0 == observed.classes1)
# [1] 0.7759197

p<-predict(m0,test)
predicted.classes <- ifelse(p > 0.5, "1", "0")
observed.classes <- horse[300:366,]$outcome
mean(predicted.classes == observed.classes)
# [1] 0.6567164

pp<-predict(m1,test)
ppredicted.classes <- ifelse(pp > 0.5, "1", "0")
observed.classes <- horse[300:366,]$outcome
mean(ppredicted.classes == observed.classes)
# [1] 0.7014925

conf2 <- as.data.frame(prop.table(table(ppredicted.classes,observed.classes),2))
ggplot(conf2) + geom_tile(aes(x=observed.classes,y=ppredicted.classes, fill=Freq)) + guides(fill=F)+
  scale_y_discrete(name="Actual Class") + scale_x_discrete(name="Predicted Class")+
  geom_text(aes(y=ppredicted.classes,x=observed.classes, label=round(Freq,3)), color="black") + 
  scale_fill_distiller(palette="Blues",direction = 1)

