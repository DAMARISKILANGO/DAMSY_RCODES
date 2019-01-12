
library(class)
library(caret)
require(mlbench)
library(e1071)
library(base)
require(base)
data(Sonar)
head(Sonar)
summary(Sonar)
str(Sonar)
cat("number of rows and columns are:", nrow(Sonar), ncol(Sonar))
    
###############
#Lets check how many ???? classes and ????

#classes Sonar data contain?and check whether Sonar data contains any NA in its columns.
#################

base::table(Sonar$Class)
apply(Sonar,2,function (x) sum(is.na(x)))

##############
# Take samples from our data to split Sonar into training 
# and test sets
##########################################

SEED<-123
set.seed(SEED)
data<-Sonar[base::sample(nrow(Sonar)),] #shuffle the data
bound<-floor(0.7*nrow(data))
df_train<-data[1:bound,]
df_test <-data[(bound + 1):nrow(data), ]

###################################33
# Let's examine if the train and test samples have properly splitted with the almost the same portion of Class labels
##################################

cat("number of training and test samples are ", nrow(df_train), nrow(df_test))

cat("number of training classes: \n", base::table(df_train$Class)/nrow(df_train))
cat("\n")
cat("number of test classes: \n", base::table(df_test$Class)/nrow(df_test))

#####################3
# To simplify our job, we can create the following data frames
#####################
X_train <- subset(df_train, select=-Class)
y_train <- df_train$Class
X_test <- subset(df_test, select=-Class) # exclude Class for prediction
y_test <- df_test$Class

############33
#Training a model on data
####################
#Now, we are going to use knn function from class library with k=3???3???=

############

model_knn <- knn(train=X_train,
                 test=X_test,
                 cl=y_train,  # class labels
                 k=3)
model_knn

####################3
#we can see how many classes have been correctly or incorrectly 
#classified by comparing to the true labels as follows
#####################

conf_mat <- base::table(y_test, model_knn)
conf_mat


##################
#T o compute the accuracy, we sum up all the correctly classified 
#observations (located in diagonal) and divide it by the total number of classes
###########

cat("Test accuracy: ", sum(diag(conf_mat))/sum(conf_mat))


########################
# To assess whether ????=3 is a good choice and see whether ????=3 leads to overfitt
# ing /underfitting the data, we could use knn.cv which does the leave-one-out 
#cross-validations for training set (i.e., it singles out a training sample one at a 
#time and tries to view it as a new example and see what class label it assigns).
#################################################################################
knn_loocv <- knn.cv(train=X_train, cl=y_train, k=3)
knn_loocv

######################
#Lets create a confusion matrix to compute the accuracy of the training 
#labels y_train and the cross-validated predictions knn_loocv, same as the above. 
#What can you find from comparing the LOOCV accuracy and the test accuracy above?
#########################################
conf_mat_cv <- base::table(y_train, knn_loocv)
conf_mat_cv
cat("LOOCV accuracy: ", sum(diag(conf_mat_cv)) / sum(conf_mat_cv))

###########################
#The difference between the cross-validated accuracy and the test accuracy shows 
#that, k????=3 leads to overfitting. Perhaps we should change ???? to lessen 
# the overfitting.
########################

##################
#Improve the performance of the model
#########################

###############3
#Cross Validation
###############

SEED <- 2016
set.seed(SEED)
# create the training data 70% of the overall Sonar data.
in_train <- createDataPartition(Sonar$Class, p=0.7, list=FALSE) # create training indices
ndf_train <- Sonar[in_train, ]
ndf_test <- Sonar[-in_train, ]

