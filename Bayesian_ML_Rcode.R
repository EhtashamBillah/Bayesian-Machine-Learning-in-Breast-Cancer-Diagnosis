

getDoParWorkers()
getDoParRegistered()
getDoParName()
getDoParVersion()
cl <- makeCluster(spec=2, type="SOCK")   # Setting up clusters for Parallel Computing
registerDoSNOW(cl)                       # Registering clusters
stopCluster(cl)                          # To stop Parallel Computing

#REQUIRED PACKAGES
require(lattice)    # for visualization
require(klaR)       # for naive bayes
require(caret)      # fitting model by k-fold cross validation
require(corrplot)   # for correlation matrix
require(ggplot2)    # for visualization
require(Amelia)     # for finding missing value
require(e1071)      # for measuring skewness
require(doSNOW)     # for parallel computation
require(ROCR)       # for ROC curve
require(glmnet)     # for variable selection with LASSO


#Loading the Dataset
dataset<- read.csv("data.csv")
dataset <- dataset[,-c(1,33)]


#########################################################
## Exploratory Data Analysis (EDA)
#########################################################

# 1.Finding missing value
missmap(dataset,
        col=c("#cc6600","#aae4f5"),
        main="Missing value Identification")

# 2.Measuring skewness of the data
skewness <- data.frame(Skewness=round((apply(dataset[,-1],2,skewness)),3))
write.csv(skewness, file= "skewness.csv")

# 3.Multicollienarity
correlation<- cor(dataset[,-1])
corr_plot <- corrplot(correlation,
                      method="color",
                      order="hclust")


# 4.Finding the outliers 
outliers <- data.frame(OutlierS=outlier(dataset[,-1]))
write.csv(outliers, file= "outliers.csv")


# An example of how spatial sign works
two_predictors<- as.data.frame(scale(final_dataset[,c("texture_worst","radius_worst")]))
xyplot(texture_worst ~ radius_worst,
       data = two_predictors,
       groups = dataset$diagnosis, 
       auto.key = list(columns = 2))

ss <- spatialSign(two_predictors)
ss <- as.data.frame(ss)
xyplot(texture_worst ~ radius_worst,
       data = ss,
       groups = dataset$diagnosis, 
       auto.key = list(columns = 2)) 


# Distribution of Dependent Variable Diagnosis
prop.table(table(dataset$diagnosis))


############################################################
# Variable selection: By Recursive Feature Elimination (RFE)
###########################################################

str(nbFuncs)
func<- nbFuncs

index <- createMultiFolds(dataset$diagnosis,times=5)
varsize <- seq(1,30,by=2)
func$summary <- function(...) c(twoClassSummary(...),
                                defaultSummary(...))

varctrl <- rfeControl(method = "repeatedcv",
                      repeats = 5,
                      verbose = TRUE,
                      functions = func,
                      index = index)

set.seed(12345)
nbrfe <- rfe(x=dataset[,-1],
             y=dataset$diagnosis,
             sizes = varsize,
             metric = "ROC",
             rfeControl = varctrl)


opt_var <- predictors(nbrfe) # gives the variables with optimum effect
nbrfe$fit
nbrfe$resample
summary(nbrfe$resample)


#Plotting the result
trellis.par.set(caretTheme())
plot(nbrfe, type = c("g", "o"),
     main= "Number of Variables Vs ROC",
     xlab="Number of Variables",
     col="green")
densityplot(nbrfe)
histogram(nbrfe)

# Taking the subset of variables that gives the optimum ROC with RFE
dataset_rfe <- dataset[,c("diagnosis",opt_var)]



###########################################
# Splitting the dataset
###########################################
# Dataset with Variables Seleted By RFE

split_rfe<- createDataPartition(y=dataset_rfe$diagnosis,p=0.70,list=FALSE)
training_set_rfe <-dataset_rfe[split_rfe,]
test_set_rfe <- dataset_rfe[-split_rfe,]
prop.table(table(training_set_rfe$diagnosis))
prop.table(table(test_set_rfe$diagnosis))




###########################################
# Variable selection using LASSO
###########################################
fit_lasso <- glmnet(x=as.matrix(dataset[,-1]), 
                    y=factor(dataset$diagnosis),
                    family="binomial",
                    standardize=TRUE,
                    alpha=1)
plot(fit_lasso,xvar="lambda",label=TRUE)

# Variable selection through cross validation
cv_lasso <- cv.glmnet(x=as.matrix(dataset[,-1]), 
                      y=factor(dataset$diagnosis),
                      family="binomial",
                      standardize=TRUE,
                      alpha=1,
                      nfolds=15,
                      type.measure="class",
                      parallel= TRUE)

plot(cv_lasso)
coef<-coef(cvfit,s='lambda.min',exact=TRUE)
index<-which(coef!=0)
optimum_subset<-row.names(coef)[index]
optimum_subset<-c(optimum_subset[-1])
dataset_lasso <- dataset[,c("diagnosis",optimum_subset)]



###########################################
# Splitting the dataset
###########################################
# Dataset with Variables Seleted By LASSO

split_lasso<- createDataPartition(y=dataset_lasso$diagnosis,p=0.70,list=FALSE)
training_set_lasso <-dataset_lasso[split,]
test_set_lasso <- dataset_lasso[-split,]
prop.table(table(training_set_lasso$diagnosis))
prop.table(table(test_set_lasso$diagnosis))



############################################################
# Bayesian Machine Learning with Variables selected by RFE
###########################################################

# Setting up control
control <- trainControl(method="repeatedcv",
                        number = 10,
                        repeats = 10,
                        classProbs = T,
                        summaryFunction = twoClassSummary,
                        allowParallel = T)

# Using normal density
grid_rfe_normal <- expand.grid(usekernel="FALSE",fL=c(0,1,2,3))
set.seed(12345)
model_bayes_cross_rfe_normal <- train(form=diagnosis~.,
                                      data=training_set_rfe,
                                      method="nb",
                                      preProcess=c("YeoJohnson","center","scale","spatialSign"),
                                      metric="ROC",
                                      trControl=control,
                                      tunegrid=grid_rfe_normal)

model_bayes_cross_rfe_normal$finalModel
importance_rfe_normal <- varImp(model_bayes_cross_rfe_normal)
plot(importance_rfe_normal,col="#4f1283",main="Normal Density (RFE)")


# Using non-parametric kernel density
# Parameter tunning for non-parametric kernel density estimation
grid_rfe_gaussian <- expand.grid(usekernel=c("TRUE","FALSE"),
                                 fL=c(0,1,2),
                                 adjust=c(1,2,3))

set.seed(12345)
model_bayes_cross_rfe_guassian <- train(form=diagnosis~.,
                                        data=training_set_rfe,
                                        method="nb",
                                        preProcess=c("YeoJohnson","center","scale","spatialSign"), 
                                        metric="ROC",
                                        trControl=control,
                                        tuneGrid=grid_rfe_gaussian)

importance_rfe_gaussian <- varImp(model_bayes_cross_rfe_guassian)
plot(importance_rfe_gaussian,col="#7add52",main="Kernel Denasity (RFE)")



############################
# Prediction
###########################

# a. Prediction ( when normal density was applied)
y_hat_rfe_normal <- predict(model_bayes_cross_rfe_normal,newdata = test_set_rfe[,-1])
cm_rfe_normal<- confusionMatrix(data=y_hat_rfe_normal,
                                reference = test_set_rfe[,1],
                                positive = "B",
                                dnn=c("PREDICTED CLASS","ACTUAL CLASS"))

# b. Prediction ( when non-parametric kernel density was applied)
y_hat_rfe_gaussian<- predict(model_bayes_cross_rfe_guassian,newdata = test_set_rfe[,-1])
cm_rfe_gaussian<- confusionMatrix(data=y_hat_rfe_gaussian,
                                  reference = test_set_rfe[,1],
                                  positive = "B",
                                  dnn=c("PREDICTED CLASS","ACTUAL CLASS"))



################################################################
# AUROC (Area under receiver operating charactaristics) CURVE
#################################################################

# a) when normal density was used
pred_rfe_normal <- prediction(predictions = as.numeric(y_hat_rfe_normal), 
                              labels=test_set_rfe$diagnosis)
perform_rfe_normal <- performance(pred_rfe_normal,
                                  measure = "tpr",
                                  x.measure="fpr")
plot(perform_rfe_normal,
     col="#8470ff",
     main="ROC Curve (RFE Normal)",
     xlab="1-Specificity",
     ylab="Sensitivity",
     type="o",
     lwd=1.5)
abline(a=0,b=1,col="#f268ae",lwd=1.5)
auc_rfe_normal <- performance(pred_rfe_normal,measure = "auc")
auc_rfe_normal <- auc_rfe_normal@y.values[[1]]
auc_rfe_normal <- round(auc_rfe_normal,4)
legend(.7,.3,auc_rfe_normal,title="AUROC",cex=0.6,
       border = "#8470ff",fill="#8470ff",bg="#dcf5e9",
       box.lwd=2,box.col = "#6666ff")


# b) when kernel density was used
pred_rfe_gaussian <- prediction(predictions = as.numeric(y_hat_rfe_gaussian), 
                                labels=as.numeric(test_set_rfe$diagnosis))
perform_rfe_gaussian <- performance(pred_rfe_gaussian,
                                    measure = "tpr",
                                    x.measure="fpr")
plot(perform_rfe_gaussian,
     col="#8470ff",
     main="ROC Curve (RFE Gaussian)",
     xlab="1-Specificity",
     ylab="Sensitivity",
     type="o",
     lwd=1.5)
abline(a=0,b=1,col="#f268ae",lwd=1.5)
auc_rfe_gaussian <- performance(pred_rfe_gaussian,measure = "auc")
auc_rfe_gaussian <- auc_rfe_gaussian@y.values[[1]]
auc_rfe_gaussian <- round(auc_rfe_gaussian,4)
legend(.7,.3,auc_rfe_gaussian,title="AUROC",cex=0.6,
       border = "#8470ff",fill="#8470ff",bg="#dcf5e9",
       box.lwd=2,box.col = "#6666ff")



################################################################
## Bayesian Machine learning with variableS selected by LASSO
###############################################################

################################################
# Finding multicollinearity after applying LASSO
################################################
correlation_lasso<- cor(dataset_lasso[,-1])
corr_plot_lasso <- corrplot(correlation_lasso,
                            method="color",
                            order="hclust")



################################################
# Model fitting with variable selected by LASSO
################################################

# Setting up train control
control <- trainControl(method="repeatedcv",
                        number = 10,
                        repeats = 10,
                        classProbs = T,
                        summaryFunction = twoClassSummary,
                        allowParallel = T)

# Model fitting using normal density
grid_lasso_normal <- expand.grid(usekernel="FALSE",fL=c(0,1,2,3))
set.seed(12345)
model_bayes_cross_lasso_normal <- train(form=diagnosis~.,
                                        data=training_set_lasso,
                                        method="nb",
                                        preProcess=c("YeoJohnson","center","scale","spatialSign"),
                                        metric="ROC",
                                        trControl=control,
                                        tunegrid=grid_lasso_normal)

importance_lasso_normal <- varImp(model_bayes_cross_lasso_normal)
plot(importance_lasso_normal,col="#8470ff",main="Normal Density (LASSO)")

# Model fitting using non-parametric kernel density
# Hyperparameter tunning for non-parametric kernel density estimation
grid_lasso_gaussian <- expand.grid(usekernel=c("TRUE","FALSE"),
                                   fL=c(0,1,2),
                                   adjust=c(1,2,3))

set.seed(12345)
model_bayes_cross_lasso_guassian <- train(form=diagnosis~.,
                                          data=training_set_lasso,
                                          method="nb",
                                          preProcess=c("YeoJohnson","center","scale","spatialSign"),
                                          metric="ROC",
                                          trControl=control,
                                          tuneGrid=grid_lasso_gaussian)

importance_lasso_gaussian <- varImp(model_bayes_cross_lasso_guassian)
plot(importance_lasso_gaussian,col="#910085",main="Kernel Density (LASSO)")

############################
# Prediction
###########################

# a. When normal density was used
y_hat_lasso_normal <- predict(model_bayes_cross_lasso_normal,newdata = test_set_lasso[,-1])
cm_lasso_normal<- confusionMatrix(data=y_hat_lasso_normal,
                                  reference = test_set_lasso[,1],
                                  positive = "B",
                                  dnn=c("PREDICTED CLASS","ACTUAL CLASS"))

# b.  With kernel density was used
y_hat_lasso_gaussian<- predict(model_bayes_cross_lasso_guassian,
                               newdata = test_set_lasso[,-1])
cm_lasso_gaussian<- confusionMatrix(data=y_hat_lasso_gaussian,
                                    reference = test_set_lasso[,1],
                                    positive = "B",
                                    dnn=c("PREDICTED CLASS","ACTUAL CLASS"))


################################################################
# AUROC (Area under receiver operating charactaristics) CURVE
#################################################################
# a) when normal density was used

pred_lasso_normal <- prediction(predictions = as.numeric(y_hat_lasso_normal), 
                                labels=test_set_lasso$diagnosis)
perform_lasso_normal <- performance(pred_lasso_normal,
                                    measure = "tpr",
                                    x.measure="fpr")
plot(perform_lasso_normal,
     col="#8470ff",
     main="ROC Curve (LASSO Normal)",
     xlab="1-Specificity",
     ylab="Sensitivity",
     type="o",
     lwd=1.5)
abline(a=0,b=1,col="#f268ae",lwd=1.5)
auc_lasso_normal <- performance(pred_lasso_normal,measure = "auc")
auc_lasso_normal <- auc_lasso_normal@y.values[[1]]
auc_lasso_normal <- round(auc_lasso_normal,4)
legend(.7,.3,auc_lasso_notmal,title="AUROC",cex=0.6,
       border = "#8470ff",fill="#8470ff",bg="#dcf5e9",
       box.lwd=2,box.col = "#6666ff")


# b) when kernel density was used

pred_lasso_gaussian <- prediction(predictions = as.numeric(y_hat_lasso_gaussian), 
                                  labels=as.numeric(test_set_lasso$diagnosis))
perform_lasso_gaussian <- performance(pred_lasso_gaussian,
                                      measure = "tpr",
                                      x.measure="fpr")
plot(perform_lasso_gaussian,
     col="#8470ff",
     main="ROC Curve (LASSO Gaussian)",
     xlab="1-Specificity",
     ylab="Sensitivity",
     type="o",
     lwd=1.5)
abline(a=0,b=1,col="#f268ae",lwd=1.5)
auc_lasso_gaussian <- performance(pred_lasso_gaussian,measure = "auc")
auc_lasso_gaussian <- auc_lasso_gaussian@y.values[[1]]
auc_lasso_gaussian <- round(auc_lasso_gaussian,4)
legend(.7,.3,auc_lasso_gaussian,title="AUROC",cex=0.6,
       border = "#8470ff",fill="#8470ff",bg="#dcf5e9",
       box.lwd=2,box.col = "#6666ff")



######################################################################
# Model Fitting with raw data (without data preprocessing)
######################################################################

#######################################################
#######################################################
# A) RFE variables set
# Setting up train control
control <- trainControl(method="repeatedcv",
                        number = 10,
                        repeats = 10,
                        classProbs = T,
                        summaryFunction = twoClassSummary,
                        allowParallel = T)

#  Using normal density
grid_rfe_normal <- expand.grid(usekernel="FALSE",fL=c(0,1,2,3))
set.seed(12345)
model_bayes_cross_rfe_normal_raw <- train(form=diagnosis~.,
                                          data=training_set_rfe,
                                          method="nb",
                                          metric="ROC",
                                          trControl=control,
                                          tunegrid=grid_rfe_normal)

importance_rfe_normal_raw <- varImp(model_bayes_cross_rfe_normal_raw)
plot(importance_rfe_normal_raw,col="#4f1283",main="Normal Density (RFE)")


# Using non-parametric kernel density
# Parameter tunning for non-parametric kernel density estimation
grid_rfe_gaussian <- expand.grid(usekernel=c("TRUE","FALSE"),
                                 fL=c(0,1,2),
                                 adjust=c(1,2,3))

set.seed(12345)
model_bayes_cross_rfe_guassian_raw <- train(form=diagnosis~.,
                                            data=training_set_rfe,
                                            method="nb",
                                            metric="ROC",
                                            trControl=control,
                                            tuneGrid=grid_rfe_gaussian)


importance_rfe_gaussian <- varImp(model_bayes_cross_rfe_guassian)
plot(importance_rfe_gaussian,col="#7add52",main="Kernel Denasity (RFE)")



############################################################
############################################################
# B)with LASSO variable set
# Setting up train control
control <- trainControl(method="repeatedcv",
                        number = 10,
                        repeats = 10,
                        classProbs = T,
                        summaryFunction = twoClassSummary,
                        allowParallel = T)

# Model fitting using normal density
grid_lasso_normal_raw <- expand.grid(usekernel="FALSE")
set.seed(12345)
model_bayes_cross_lasso_normal_raw <- train(form=diagnosis~.,
                                            data=training_set_lasso,
                                            method="nb",
                                            metric="ROC",
                                            trControl=control,
                                            tunegrid=grid_lasso_normal_raw)

importance_lasso_normal_raw <- varImp(model_bayes_cross_lasso_normal_raw)
plot(importance_lasso_normal_raw,col="#8470ff",main="Normal Density Raw (LASSO)")

# Model fitting using non-parametric kernel density
# Hyperparameter tunning for non-parametric kernel density estimation
grid_lasso_gaussian_raw <- expand.grid(usekernel=c("TRUE","FALSE"),
                                       fL=c(0,1,2),
                                       adjust=c(1,2,3))

set.seed(12345)
model_bayes_cross_lasso_guassian_raw <- train(form=diagnosis~.,
                                              data=training_set_lasso,
                                              method="nb",
                                              metric="ROC",
                                              trControl=control,
                                              tuneGrid=grid_lasso_gaussian_raw)
importance_lasso_gaussian_raw <- varImp(model_bayes_cross_lasso_guassian_raw)
plot(importance_lasso_gaussian_raw,col="#910085",main="Kernel Density Raw (LASSO)")



############################
# Prediction
###########################

#####################################################
#####################################################
# With RFE dataset
# a)Prediction by model with normal density
y_hat_rfe_normal_raw <- predict(model_bayes_cross_rfe_normal_raw,newdata = test_set_rfe[,-1])
cm_rfe_normal_raw<- confusionMatrix(data=y_hat_rfe_normal_raw,
                                    reference = test_set_rfe[,1],
                                    positive = "B",
                                    dnn=c("PREDICTED CLASS","ACTUAL CLASS"))

# b) Prediction by model with kernel density
y_hat_rfe_gaussian_raw<- predict(model_bayes_cross_rfe_guassian_raw,newdata = test_set_rfe[,-1])
cm_rfe_gaussian_raw<- confusionMatrix(data=y_hat_rfe_gaussian_raw,
                                      reference = test_set_rfe[,1],
                                      positive = "B",
                                      dnn=c("PREDICTED CLASS","ACTUAL CLASS"))

##################################################
##################################################



#################################################
#################################################
# With LASSO dataset
# a. with normal density
y_hat_lasso_normal_raw <- predict(model_bayes_cross_lasso_normal_raw,newdata = test_set_lasso[,-1])
cm_lasso_normal_raw<- confusionMatrix(data=y_hat_lasso_normal_raw,
                                      reference = test_set_lasso[,1],
                                      positive = "B",
                                      dnn=c("PREDICTED CLASS","ACTUAL CLASS"))

# b.  with kernel density
y_hat_lasso_gaussian_raw<- predict(model_bayes_cross_lasso_guassian_raw,
                                   newdata = test_set_lasso[,-1])
cm_lasso_gaussian_raw<- confusionMatrix(data=y_hat_lasso_gaussian_raw,
                                        reference = test_set_lasso[,1],
                                        positive = "B",
                                        dnn=c("PREDICTED CLASS","ACTUAL CLASS"))

##################################################
##################################################





###############################
# AUROC
##############################

##################################################
##################################################
# with RFE dataset
# a) When normal density was used
pred_rfe_normal_raw <- prediction(predictions = as.numeric(y_hat_rfe_normal_raw), 
                                  labels=test_set_rfe$diagnosis)
perform_rfe_normal_raw <- performance(pred_rfe_normal_raw,measure = "tpr",x.measure="fpr")
plot(perform_rfe_normal_raw,
     col="#8470ff",
     main="ROC Curve (RFE Normal)",
     xlab="1-Specificity",
     ylab="Sensitivity",
     type="o",
     lwd=1.5)
abline(a=0,b=1,col="#f268ae",lwd=1.5)
auc_rfe_normal_raw <- performance(pred_rfe_normal_raw,measure = "auc")
auc_rfe_normal_raw <- auc_rfe_normal_raw@y.values[[1]]
auc_rfe_normal_raw <- round(auc_rfe_normal_raw,4)
legend(.7,.3,auc_rfe_normal_raw,title="AUROC",cex=0.6,
       border = "#8470ff",fill="#8470ff",bg="#dcf5e9",
       box.lwd=2,box.col = "#6666ff")


# b) when kernel density was used

pred_rfe_gaussian_raw <- prediction(predictions = as.numeric(y_hat_rfe_gaussian_raw), 
                                    labels=as.numeric(test_set_rfe$diagnosis))
perform_rfe_gaussian_raw <- performance(pred_rfe_gaussian_raw,measure = "tpr",x.measure="fpr")
plot(perform_rfe_gaussian_raw,
     col="#8470ff",
     main="ROC Curve (RFE Gaussian)",
     xlab="1-Specificity",
     ylab="Sensitivity",
     type="o",
     lwd=1.5)
abline(a=0,b=1,col="#f268ae",lwd=1.5)
auc_rfe_gaussian_raw <- performance(pred_rfe_gaussian_raw,measure = "auc")
auc_rfe_gaussian_raw <- auc_rfe_gaussian_raw@y.values[[1]]
auc_rfe_gaussian_raw <- round(auc_rfe_gaussian_raw,4)
legend(.7,.3,auc_rfe_gaussian_raw,title="AUROC",cex=0.6,
       border = "#8470ff",fill="#8470ff",bg="#dcf5e9",
       box.lwd=2,box.col = "#6666ff")

######################################################
######################################################


######################################################
######################################################

# With LASSO dataset
# a) When normal density was used
pred_lasso_normal_raw <- prediction(predictions = as.numeric(y_hat_lasso_normal_raw), 
                                    labels=as.numeric(test_set_lasso$diagnosis))
perform_lasso_normal_raw <- performance(pred_lasso_normal_raw,
                                        measure = "tpr",
                                        x.measure="fpr")
plot(perform_lasso_normal_raw,
     col="#8470ff",
     main="ROC Curve (LASSO Normal)",
     xlab="1-Specificity",
     ylab="Sensitivity",
     type="o",
     lwd=1.5)

abline(a=0,b=1,col="#f268ae",lwd=1.5)
auc_lasso_normal_raw <- performance(pred_lasso_normal_raw,measure = "auc")
auc_lasso_normal_raw <- auc_lasso_normal_raw@y.values[[1]]
auc_lasso_normal_raw <- round(auc_lasso_normal_raw,4)
legend(.7,.3,auc_lasso_normal_raw,title="AUROC",cex=0.6,
       border = "#8470ff",fill="#8470ff",bg="#dcf5e9",
       box.lwd=2,box.col = "#6666ff")


# b) When kernel density was used

pred_lasso_gaussian_raw <- prediction(predictions = as.numeric(y_hat_lasso_gaussian_raw), 
                                      labels=as.numeric(test_set_lasso$diagnosis))
perform_lasso_gaussian_raw <- performance(pred_lasso_gaussian_raw,
                                          measure = "tpr",
                                          x.measure="fpr")

plot(perform_lasso_gaussian_raw,
     col="#8470ff",
     main="ROC Curve(Lasso Guassian)",
     xlab="1-Specificity",
     ylab="Sensitivity",
     type="o",
     lwd=1.5)

abline(a=0,b=1,col="#f268ae",lwd=1.5)
auc_lasso_gaussian_raw <- performance(pred_lasso_gaussian_raw,measure = "auc")
auc_lasso_gaussian_raw <- auc_lasso_gaussian_raw@y.values[[1]]
auc_lasso_gaussian_raw <- round(auc_lasso_gaussian_raw,4)
legend(.7,.3,auc_lasso_gaussian_raw,title="AUROC",cex=0.6,
       border = "#8470ff",fill="#8470ff",bg="#dcf5e9",
       box.lwd=2,box.col = "#6666ff")
#########################################################
#########################################################



##################################################
# Summarizing all models and Predictions
##################################################

# All models
model_bayes_cross_rfe_normal
model_bayes_cross_rfe_guassian
model_bayes_cross_lasso_normal
model_bayes_cross_lasso_guassian
model_bayes_cross_rfe_normal_raw
model_bayes_cross_rfe_guassian_raw
model_bayes_cross_lasso_normal_raw
model_bayes_cross_lasso_guassian_raw


# All predictions
cm_rfe_normal
cm_rfe_gaussian
cm_lasso_normal
cm_lasso_gaussian
cm_rfe_normal_raw
cm_rfe_gaussian_raw
cm_lasso_normal_raw
cm_lasso_gaussian_raw


# Saving All Models
saveRDS(model_bayes_cross_rfe_normal,file="bayes_normal")
saveRDS(model_bayes_cross_rfe_guassian,file="bayes_gaussian")
saveRDS(model_bayes_cross_lasso_normal,file="bayes_lasso_normal")
saveRDS(model_bayes_cross_lasso_guassian,file="bayes_lasso_gaussian")
saveRDS(model_bayes_cross_rfe_normal_raw,file="bayes_normal_raw")
saveRDS(model_bayes_cross_rfe_guassian_raw,file="bayes_gaussian_raw")
saveRDS(model_bayes_cross_lasso_normal_raw,file="bayes_lasso_normal_raw")
saveRDS(model_bayes_cross_lasso_guassian_raw,file="bayes_lasso_gaussian_raw")
