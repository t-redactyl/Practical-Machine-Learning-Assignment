# Load the data
rm(list = ls())
setwd("/Users/jburchell/Documents/Practical-Machine-Learning-Assignment")

# Download data
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "data.csv")
data <- read.csv("data.csv")

# Split data into testing and training
library(caret); library(magrittr)
set.seed(567)
inTrain <- createDataPartition(y = data$classe, p = 0.7, list = FALSE)
training <- data[inTrain, ]
testing <- data[-inTrain, ]
nrow(training)

# Download validation dataset
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", "validation.csv")
validation <- read.csv("validation.csv")
rm(list = c("data", "inTrain"))

# Examine the data
str(training)
head(training$classe)
table(training$classe)
prop.table(table(training$classe))

# Six young healthy participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell 
# Biceps Curl in five different fashions: 
    # Class A: exactly according to the specification
    # Class B: throwing the elbows to the front
    # Class C: lifting the dumbbell only halfway
    # Class D: lowering the dumbbell only halfway and 
    # Class E: throwing the hips to the front.

# The exercises were performed by six male participants aged between 20-28 years, with little weight lifting 
# experience. We made sure that all participants could easily simulate the mistakes in a safe and controlled
# manner by using a relatively light dumbbell (1.25kg).

# For data recording we used four 9 degrees of freedom Razor inertial measurement units (IMU), which 
# provide three-axes acceleration, gyroscope and magnetometer data at a joint sampling rate of 45 Hz. 
# Each IMU also featured a Bluetooth module to stream the recorded data to a notebook running the Context 
# Recognition Network Toolbox [3]. We mounted the sensors in the users’ glove, armband, lumbar belt and
# dumbbell (see Figure 1). We designed the tracking system to be as unobtrusive as possible, as these are 
# all equipment commonly used by weight lifters. 

# Convert all of the factor predictor variables to numeric

asNumeric <- function(x) as.numeric(as.character(x))
factorsNumeric <- function(d) modifyList(d, lapply(d[, sapply(d, is.factor)], 
                                                asNumeric))
training %>%
    subset(select = -c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, 
                      cvtd_timestamp, new_window, num_window, classe)) %>%
    factorsNumeric -> training[ , 8:159]
    
# It appears that a number of variables have missing data. 

propmiss <- function(dataframe) {
    sapply(dataframe,function(x) 
    data.frame(nmiss=sum(is.na(x)), 
               n=length(x), 
               propmiss=sum(is.na(x))/length(x)))
}

# Get rid of variable with more than 5% missing (which turns out to be variables with any missing, as
# missing data is around 90% for these variables)
propmiss(training)
sum(sapply(training, function(x) sum(is.na(x))/length(x) >= 0.05))
training <- training[, sapply(training, function(x) sum(is.na(x))/length(x) < 0.05)]

# Split training into training and calibration
inTrain <- createDataPartition(y = training$classe, p = 0.75, list = FALSE)
tTraining <- training[inTrain, ]
tCalibration <- training[-inTrain, ]


require(ggplot2); library(scales)

plots <- function(yvar) {
    qplot(cvtd_timestamp, yvar, data = training1, geom = "line") %>%
        add(geom_line(aes(x=cvtd_timestamp, y=roll_forearm, colour=user_name))) %>% 
        add(xlab('')) %>%
        add(ylab("yvar")) %>%
        add(theme_bw())    
}

lapply(training1[ , 8:9], plots)

propmiss(training1)

vars_left <- data.frame(Device <- character(length = 52), 
                        Predictor <- colnames(training1[8:59]))
names(vars_left) <- c("Device", "Predictor")
vars_left$Device <- as.character(vars_left$Device)
vars_left$Predictor <- as.character(vars_left$Predictor)

vars_left$Device[grepl("belt", vars_left$Predictor, ignore.case = TRUE)] <- "Belt"
vars_left$Device[grepl("arm", vars_left$Predictor, ignore.case = TRUE)] <- "Arm"
vars_left$Device[grepl("dumbbell", vars_left$Predictor, ignore.case = TRUE)] <- "Dumbbell"
vars_left$Device[grepl("forearm", vars_left$Predictor, ignore.case = TRUE)] <- "Forearm"

vars_left$Device <- as.factor(vars_left$Device)
table(vars_left$Predictor, vars_left$Device)

library(knitr)
kable(vars_left)

vars_left2 <- vars_left
vars_left2$Predictor <- gsub("_belt|_arm|_dumbbell|_forearm", "", vars_left2$Predictor)
table(vars_left2$Predictor, vars_left2$Device)

# Check for near-zero variance predictors
nzv <- nearZeroVar(training1[8:59], saveMetrics= TRUE)
nzv

# Describe the distribution more explicitly
sapply(training1[ , 8:59], summary)

# Everything appears to have good variance, a couple of outliers here and there though.

# Now I have a final list of cleaned variables, I need to work out their relationships/distributions.
training1[ , 8:59] %>%
    cor %>% abs

spec.cor <- function (dat, r, ...) { 
    x <- cor(dat, ...) 
    x[upper.tri(x, TRUE)] <- NA 
    i <- which(abs(x) >= r, arr.ind = TRUE) 
    data.frame(matrix(colnames(x)[as.vector(i)], ncol = 2), value = x[i]) 
} 

spec.cor(training1[ , 8:59], 0.8)

# Exploring relationships between outcome and predictors
sapply()

lapply(training1[ , 8:59], function(x) summary(aov(x ~ classe, data = training1)))

anovas <- data.frame(Predictors <- as.character(), 
                     F_value <- as.numeric(), 
                     p_value <- as.numeric())
aov.temp <- aov(training1[ , 8] ~ classe, data = training1)
x <- c(names(training1[i]),
       summary(aov.temp)[[1]][["F value"]][[1]], 
       summary(aov.temp)[[1]][["Pr(>F)"]][[1]])
anovas <- rbind(anovas, x)

for (i in 9:59) {
    aov.temp <- aov(training1[ , i] ~ classe, data = training1)
    x <- c(names(training1[i]),
           summary(aov.temp)[[1]][["F value"]][[1]], 
           summary(aov.temp)[[1]][["Pr(>F)"]][[1]])
    levels(anovas$X.roll_belt.) <- c(levels(anovas$X.roll_belt.), names(training1[i]))
    anovas <- rbind(anovas, x)
    x <- NULL
} 
anovas

aov.temp <- aov(training1[ , 8] ~ classe, data = training1)
x <- c(names(training1[i]),
       summary(aov.temp)[[1]][["F value"]][[1]], 
       summary(aov.temp)[[1]][["Pr(>F)"]][[1]])
anovas <- rbind(anovas, x)


lapply(training1[ , 8:59], spec.cor(training1))

i <- c(x, summary(aov.temp)[[1]][["F value"]][[1]], summary(aov.temp)[[1]][["Pr(>F)"]][[1]])

data.frame(Predictor = colnames(x))

anovas <- as.data.frame(sapply(training1[ , 8:59], function(x)
    summary(aov(x ~ classe, data = training1))[[1]][["F value"]][[1]]))
names(anovas) <- "F_values"
# Top F-values from ANOVAS
head(anovas[order(-anovas$F_values),  , drop = FALSE])

d.plots <- function(predictor, label) {
    qplot(predictor, colour = classe, data = training1, geom = "density") + 
        xlab(label) +
        ylab("Density") +
        ggtitle(label) +
        theme_bw()
}

g1 <- d.plots(training1$pitch_forearm, "Forearm Pitch")
g2 <- d.plots(training1$magnet_belt_y, "Belt Magnetometer, Y-axis")
g3 <- d.plots(training1$magnet_arm_x, "Arm Magnetometer, X-axis")
g4 <- d.plots(training1$magnet_arm_y, "Arm Magnetometer, Y-axis")
g5 <- d.plots(training1$accel_arm_x, "Arm Acceleration, X-axis")
g6 <- d.plots(training1$accel_forearm_x, "Forearm Acceleration, X-axis")

g7 <- d.plots(training1$yaw_belt, "Belt Yaw")
g8 <- d.plots(training1$roll_belt, "Belt Roll")
g9 <- d.plots(training1$magnet_dumbbell_z, "Dumbbell Magnetometer, Z-axis")

qplot(roll_belt, pitch_forearm, colour = classe, data = training1) + 
    xlab("Forearm pitch") +
    ylab("Belt roll") +
    theme_bw()

library(gridExtra)
grid.arrange(g1, g2, g3, g4, g5, g6, nrow = 3, ncol = 2)

set.seed(125)
modFit <- train(classe ~ ., method = "rpart", data = training1[ , c(2, 8:60)])
fancyRpartPlot(modFit$finalModel)


install.packages('randomForest')
library(randomForest)

set.seed(567)
fit <- randomForest(classe ~ ., data = training1[ , c(2, 8:60)], importance=TRUE, ntree=2000)

varImpPlot(fit)

library(rattle)
fancyRpartPlot(fit)



col <- c("#FD8D3C", "#FD8D3C", "#FD8D3C", "#BCBDDC",
         "#FDD0A2", "#FD8D3C", "#BCBDDC")
prp(modFit$finalModel, type=2, extra=104, nn=TRUE, fallen.leaves=TRUE,
    faclen=0, varlen=0, shadow.col="grey", branch.lty=3, box.col= col)


# Notes - 
    # Working out in-sample error (predicted against observed)
    # Have a read about cross-validation techniques and estimating out-of-sample error
    # Read about strengths of different techniques with many categories for classification

user.plots <- function(predictor, label) {
    qplot(predictor, colour = user_name, data = training1, geom = "density") + 
        xlab(label) +
        ylab("Density") +
        ggtitle(label) +
        theme_bw()
}

user.plots(training1$pitch_forearm, "Forearm Pitch")

qplot(pitch_forearm, colour = user_name, 
      data = training1[training1$user_name != "adelmo", ], 
      geom = "density") + 
    xlab("Forearm Pitch") +
    ylab("Density") +
    ggtitle("Forearm Pitch") +
    theme_bw()

user.plots(training1$magnet_belt_y, "Belt Magnetometer, Y-axis")
user.plots(training1$magnet_arm_x, "Arm Magnetometer, X-axis")
user.plots(training1$magnet_arm_y, "Arm Magnetometer, Y-axis")
user.plots(training1$accel_arm_x, "Arm Acceleration, X-axis")
user.plots(training1$accel_forearm_x, "Forearm Acceleration, X-axis")
user.plots(training1$yaw_belt, "Belt Yaw")
user.plots(training1$roll_belt, "Belt Roll")
user.plots(training1$magnet_dumbbell_z, "Dumbbell Magnetometer, Z-axis")

# define training control
train_control <- trainControl(method="cv", number=10)
# train the model 
model <- train(classe ~ ., data=training1[, 8:60], trControl=train_control, method="nb")
model <- train(classe ~ ., data=training1[, 8:60], method="nb")
# make predictions
predictions <- predict(model, training1[,8:59])
# summarize results
confusionMatrix(predictions, training1$classe)$byClass

# Manually completing k-fold cross-validation
var <- "Var217"
aucs <- rep(0, 100)
for (rep in 1:length(aucs)) {
    useForCalRep <- rbinom(n = dim(dTrainAll)[[1]], size = 1, prob = 0.1) > 0
    predRep <- mkPredC(dTrainAll[!useForCalRep, outcome],
                       dTrainAll[!useForCalRep, var],
                       dTrainAll[useForCalRep, var])
    aucs[rep] <- calcAUC(predRep, dTrainAll[useForCalRep, outcome])
}

mean(aucs)
sd(aucs)

folds <- createFolds(training1$classe, k = 10, list = TRUE, returnTrain = FALSE)
model <- train(classe ~ ., data=training1[, 8:60], method="nb")

knn(train = x[ -idx[[1]], ], test = x[ idx[[1]], ], cl=t$Tissue[ -idx[[1]] ], k=5)

# Practical Data Science in R - Memorisation Models

# Check the class of all predictors
sapply(training[,8:59], class)

# Building single-variable models
## Plotting churn grouped by variable 218 levels

# Step 1: Test highest predicting variables (test and calibration) using multinomial logistic regression
# Step 2: Using reduced list, test decision tree, k-nearest neighbours and Naive Bayes models
# Step 3: Using cross-validation to get mean and SD of accuracy or AUC? for each of these models

require(nnet)

tTraining$classe2 <- relevel(tTraining$classe, ref = "A")
tCalibration$classe2 <- relevel(tCalibration$classe, ref = "A")

predictors <- names(training[ , 8:59])
outcome <- "classe"

test <- multinom(classe2 ~ tTraining[ , 8], data = tTraining)
summary(test)
pred_test <- predict(reg5, newdata = test, type="response")

folds <- createFolds(training$classe, k = 10, list = TRUE, returnTrain = TRUE)
modelCV <- function(k, method) {
    model <- train(classe ~ ., data = training[folds[[k]], 8:60], methods = method)
    predictions <- predict(model, training[-folds[[k]],8:59])
    confusionMatrix(predictions, training[folds[[k]], 8:60]$classe)
}


folds <- createFolds(training$classe, k = 10, list = TRUE, returnTrain = TRUE)
model <- train(classe ~ ., data=training[folds[[1]], 8:60], method="nb")
predictions <- predict(model, training[-folds[[1]],8:59])
confusionMatrix(predictions, training[folds[[1]], 8:60]$classe)$byClass

folds <- createFolds(training$classe, k = 10, list = TRUE, returnTrain = TRUE)
accuracies.dt <- c()
for (i in 1:10) {
    model <- train(classe ~ ., data=training[folds[[i]], 8:60], method = "rpart")
    predictions <- predict(model, training[-folds[[i]],8:59])
    accuracies.dt <- c(accuracies, 
                       confusionMatrix(predictions, training[-folds[[i]], 8:60]$classe)$overall[[1]])
}

accuracies.rf <- c()
for (i in 1:10) {
    model <- train(classe ~ ., data=training[folds[[i]], 8:60], method = "rf")
    predictions <- predict(model, training[-folds[[i]],8:59])
    accuracies.rf <- c(accuracies, 
                       confusionMatrix(predictions, training[-folds[[i]], 8:60]$classe)$overall[[1]])
}

accuracies.nb <- c()
for (i in 1:10) {
    model <- train(classe ~ ., data=training[folds[[i]], 8:60], method = "nb")
    predictions <- predict(model, training[-folds[[i]],8:59])
    accuracies.nb <- c(accuracies, 
                       confusionMatrix(predictions, training[-folds[[i]], 8:60]$classe)$overall[[1]])
}

folds <- createFolds(training$classe, k = 10, list = TRUE, returnTrain = TRUE)
model <- train(classe ~ ., data=training[folds[[1]], 8:60], method="rf")
predictions <- predict(model, training[-folds[[1]],8:59])
confusionMatrix(predictions, training[-folds[[1]], 8:60]$classe)










