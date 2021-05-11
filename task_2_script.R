
#Set my WD and seed for convenience
setwd("~/Documents/Coding/CFAS420/Data")
set.seed(1)
#Loading data - two data frames were creatd, tuned and imagenet
non_c_imgenet <- read.csv('NON_COVID_imagenet.csv')
c_imgenet <- read.csv('COVID_imagenet.csv')
non_c_tuned <- read.csv('NON_COVID_tuned.csv')
c_tuned <- read.csv('COVID_tuned.csv')
c_tuned <- na.omit(c_tuned)
imagenet <- rbind(non_c_imgenet, c_imgenet)
tuned <- rbind(non_c_tuned, c_tuned)

#appropriate libraries loaded
library(vcd)
library(ISLR)
library(skimr)
library(rpart)
library(rattle)
library(dplyr)
library(bst)# PCA
library(caret)
library(ggfortify)

#Manual scale function
scalar1 <- function(x) {(x - mean(x))/sd(x)}

#Manually scale all cols that arent 0 columns
for (col in 2:4097){
  if (all(imagenet[,col]==0) == F){
    imagenet[,col] <- scalar1(imagenet[,col])
  }
}


#tuned <- tuned[, colSums(tuned != 0) > 0]
#imagenet <- imagenet[, colSums(imagenet != 0) > 0]

#PCA on imagenet data
pca_imagenet <- prcomp(imagenet[,2:4097], scale = F, rank = 10)

#Percent of variance explained and plotted
pcent_var_expln <- 100*pca_imagenet$sdev^2/sum(pca_imagenet$sdev^2)
plot(pcent_var_expln[1:10], type='o', xlab = 'Principal Component',
     ylab = 'Percentage of variance explained',
     main = 'Percentage of variance explained for the first 10 PC\'s (Imagenet)' )
reduced_data <- as.data.frame(pca_imagenet$x)

#Outcomes created as factor variable
outcomes <- c(rep(0, 250), rep(1, 250))
outcomes <- as.factor(outcomes)
imagenet$outcomes <- as.factor(outcomes)

#Plotting the relationship between pc1 and pc2
ggplot(data = pca_imagenet, aes(x=PC1, y=PC2, colour=outcomes)) +geom_point() +
  labs(title = 'The relationship between PC1 and PC2 for the Imagenet data')


#PCA for Tuned - the code is the same as above

for (col in 2:4097){
  if (all(tuned[,col]==0) == F){
    tuned[,col] <- scalar1(tuned[,col])
  }
}

pca_tuned <- prcomp(tuned[,2:4097], scale = F, rank = 10)
pcent_var_expln_tune <- 100*pca_tuned$sdev^2/sum(pca_tuned$sdev^2)
plot(pcent_var_expln_tune[1:10], type='o', xlab = 'Principal Component', col = 'red',
     ylab = 'Percentage of variance explained',
     main = 'PVE for the first 10 Principal Components' )
points(pcent_var_expln[1:10], type = 'o', col = 'blue')
legend(8, 20, legend = c('Imagenet', 'Tuned'), col = c('blue', 'red'), 
pch =19, cex=0.8, title = 'Data')


sum(pcent_var_expln_tune[1:10])
sum(pcent_var_expln[1:10])


ggplot(pca_tuned, aes(x=PC1, y=PC2, colour=outcomes)) +geom_point() +
  labs(title = 'The relationship between PC1 and PC2 for the Tuned data')



#CLASSIFICATION PART TWO
#binding the results (outcome) to the two dataframes
outcomes <- c(rep(0, 250), rep(1, 250))
imagenet <- rbind(non_c_imgenet, c_imgenet)
imagenet <- cbind(imagenet, outcomes)
tuned <- rbind(non_c_tuned, c_tuned)

tuned <- cbind(tuned, outcomes)

#splitting data
train_indicies <- createDataPartition(tuned$outcomes,
                                      p=0.8,
                                      list = FALSE,
                                      times = 1)
#The same split was applied to both datasets
tuned_train <- tuned[train_indicies,]
tuned_test <- tuned[-train_indicies,]

imagenet_train <- imagenet[train_indicies,]
imagenet_test <- imagenet[-train_indicies,]
tuned <- rbind(non_c_tuned, c_tuned)


#LASSO LOG REG
library(glmnet)
library(stepPlr)
library(LogicReg)
library(pROC)

#tuned data first - needs to be in a matrix form
tuned_matrix <- model.matrix(outcomes~., tuned_train[,2:4098])[,-1]

# Convert the outcome (class) to a numerical variable
tuned_train$outcomes <- as.factor(tuned_train$outcomes)
levels(tuned_train$outcomes)[levels(tuned_train$outcomes)== '0'] <- 'Covid Negative'
levels(tuned_train$outcomes)[levels(tuned_train$outcomes)== '1'] <- 'Covid Positive'
tuned_response <- ifelse(tuned_train$outcomes == "Covid Positive", 1, 0)

#Finding the minimum lambda value
cv.tuned <- cv.glmnet(tuned_matrix, tuned_response, alpha = 1, family = "binomial")
plot(cv.tuned)
title('Grid search for optimal Lambda for tuned data', line = 3)


sum(coef(cv.tuned, cv.tuned$lambda.min) ==0) # when lambda.min, 4062 are 0, when 1se, 4065 are 0 

#training the model using cross validation
trctrl <- trainControl(method = "cv", number = 5)
#1se lambda was used
tuningGrid_tuned <- expand.grid(lambda = cv.tuned$lambda.1se, alpha = 1)
log_moded_tuned <- caret::train(outcomes ~., data = tuned_train[,2:4098], method = "glmnet",
                                tuneGrid = tuningGrid_tuned,
                                trControl=trctrl)

#Predicting the tuned data
log_pred_tuned <- predict(log_moded_tuned, newdata = tuned_test[,2:4097], type = 'prob')

#Creating an roc curve using the predictions and actual outcomes
roc_tuned <- roc(as.integer(tuned_test$outcomes)-1, unlist(log_pred_tuned[2]))

#plotting the ROC
plot(roc_tuned, col = 'red', main = 'ROC curve for tuned data')

#seeing which threshold returns the best accuracy
coords(roc_tuned, x='best', transpose = T, ret = c('sensitivity','specificity',
                                                    'accuracy', 'threshold'))


#Imagenet logistic - alot of the code is repeated from above
img_matrix <- model.matrix(outcomes~., imagenet_train[,2:4098])[,-1]
# Convert the outcome (class) to a numerical variable
imagenet_train$outcomes <- as.factor(imagenet_train$outcomes)
levels(imagenet_train$outcomes)[levels(imagenet_train$outcomes)== '0'] <- 'Covid Negative'
levels(imagenet_train$outcomes)[levels(imagenet_train$outcomes)== '1'] <- 'Covid Positive'
image_response <- ifelse(imagenet_train$outcomes == "Covid Positive", 1, 0)

#finding the optimal lambda for the imagenet data
cv.img <- cv.glmnet(img_matrix, image_response, alpha = 1, family = "binomial")
plot(cv.img)
title('Grid search for optimal Lambda for imagenet data', line = 3)



sum(coef(cv.img, cv.img$lambda.1se) ==0) # when lambda.min, 3986 are 0, when 1se, 4023 are 0 

#the model is trained here using the tuning grid and same training controls as before
tuningGrid_img <- expand.grid(lambda = cv.img$lambda.1se, alpha = 1)
log_moded_img <- caret::train(outcomes ~., data = imagenet_train[,2:4098], method = "glmnet",
                                tuneGrid = tuningGrid_img,
                                trControl=trctrl)

#Predicting the result as a probability
log_pred_img <- predict(log_moded_img, newdata = imagenet_test[,2:4098], type = 'prob')

#Creating the imagenet ROC curve
roc_img <- roc(as.integer(imagenet_test$outcomes)-1, unlist(log_pred_img[2]))

plot(roc_img, col = 'blue', main = 'ROC curve for Imagenet data')



#Comparing the two ROCs
plot(roc_img, col = 'blue', main = 'ROC curve for Imagenet data')
plot(roc_tuned, add = T, col = 'red')
legend(0.2, 0.3, legend = c('Tuned images', 'Imagenet'), col = c('red', 'blue'), 
       lty = 1:1 ,cex=0.8, title = 'Dataset')

#The ROCs have different bedt points
coords(roc_tuned, x='best', transpose = T, ret = c('sensitivity','specificity',
                                                   'accuracy', 'threshold'))
coords(roc_img, x='best', transpose = T, ret = c('sensitivity','specificity',
                                                   'accuracy', 'threshold'))
#AUC can be used as an accuracy metric
auc(roc_tuned)
auc(roc_img)



#FINAL TASK - clustering
library(factoextra)

#Creating a new data frame from the PCs
pca_tuned_data <- as.data.frame(pca_tuned$x)
#pca_tuned_data$outcomes <- as.factor(outcomes)


library(mclust)

# new df of first 3 PCs
pca_tuned_three <- as.data.frame(pca_tuned_data[,1:3])

#Performing the clustering on the first 3 PCs
gmm_model <-Mclust(pca_tuned_three)

#these plots produce multiple graphs, have to cycle through to find desired graph
plot(gmm_model)

pca_tuned_data$outcomes <- as.factor(outcomes)

#Relationship between pc1 and pc2
ggplot(pca_tuned_data, aes(x=PC1, y=PC2, colour=outcomes)) +geom_point() + geom_density2d(adjust = 0.5)


summary(gmm_model)

# Seeing how many clusters are optimal
fviz_mclust_bic(gmm_model)

library(pdist)

#The centre of every cluster
cluster_centres <- gmm_model[["parameters"]][["mean"]]

#This was the quickest way i found to find the neaest point to every cluster
#I made a loop for each cluster independently, 6 in total, they all return the 
# nearest subject ID, the distance from the centroid and whether they have covid or not

#looping through to find the nearest cluster to cluster 1
# the loop checks if the new subject is more similar to the centroid than the current most similar
# if it is, it becomes the new nearest subject, this is repeated for every value
cluster_1_ID <- 1000
cluster_1_dist <- 1000
cluster_1_covid <- 0
for (row in 1:500){
  distance <- pdist(pca_tuned_three[row,], cluster_centres[,1])
  distance1 <- distance@dist
  if (distance1 < cluster_1_dist){ 
    cluster_1_ID <- row
    cluster_1_dist <- distance1
    if (cluster_1_ID > 250){
      cluster_1_ID <- cluster_1_ID - 250
      cluster_1_covid <- 1
    }
  }
  cluster_1 <- c(cluster_1_ID, cluster_1_dist, cluster_1_covid)
}

#Cluster 2
cluster_2_ID <- 1000
cluster_2_dist <- 1000
cluster_2_covid <- 0
for (row in 1:500){
  distance <- pdist(pca_tuned_three[row,], cluster_centres[,2])
  distance1 <- distance@dist
  if (distance1 < cluster_2_dist){
    cluster_2_ID <- row
    cluster_2_dist <- distance1
    if (cluster_2_ID > 250){
      cluster_2_ID <- cluster_2_ID - 250
      cluster_2_covid <- 1
    }
  }
  cluster_2 <- c(cluster_2_ID, cluster_2_dist, cluster_2_covid)
}

#Cluster 3
cluster_3_ID <- 1000
cluster_3_dist <- 1000
cluster_3_covid <- 0
for (row in 1:500){
  distance <- pdist(pca_tuned_three[row,], cluster_centres[,3])
  distance1 <- distance@dist
  if (distance1 < cluster_3_dist){
    cluster_3_ID <- row
    cluster_3_dist <- distance1
    if (cluster_3_ID > 250){
      cluster_3_ID <- cluster_3_ID - 250
      cluster_3_covid <- 1
    }
  }
  cluster_3 <- c(cluster_3_ID, cluster_3_dist, cluster_3_covid)
}

#Cluster 4
cluster_4_ID <- 1000
cluster_4_dist <- 1000
cluster_4_covid <- 0
for (row in 1:500){
  distance <- pdist(pca_tuned_three[row,], cluster_centres[,4])
  distance1 <- distance@dist
  if (distance1 < cluster_4_dist){
    cluster_4_ID <- row
    cluster_4_dist <- distance1
    if (cluster_4_ID > 250){
      cluster_4_ID <- cluster_4_ID - 250
      cluster_4_covid <- 1
    }
  }
  cluster_4 <- c(cluster_4_ID, cluster_4_dist, cluster_4_covid)
}

#Cluster 5
cluster_5_ID <- 1000
cluster_5_dist <- 1000
cluster_5_covid <- 0
for (row in 1:500){
  distance <- pdist(pca_tuned_three[row,], cluster_centres[,5])
  distance1 <- distance@dist
  if (distance1 < cluster_5_dist){
    cluster_5_ID <- row
    cluster_5_dist <- distance1
    if (cluster_5_ID > 250){
      cluster_5_ID <- cluster_5_ID - 250
      cluster_5_covid <- 1
    }
  }
  cluster_5 <- c(cluster_5_ID, cluster_5_dist, cluster_5_covid)
}
#cluster 6
cluster_6_ID <- 1000
cluster_6_dist <- 1000
cluster_6_covid <- 0
for (row in 1:500){
  distance <- pdist(pca_tuned_three[row,], cluster_centres[,6])
  distance1 <- distance@dist
  if (distance1 < cluster_6_dist){
    cluster_6_ID <- row
    cluster_6_dist <- distance1
    if (cluster_6_ID > 250){
      cluster_6_ID <- cluster_6_ID - 250
      cluster_6_covid <- 1
    }
  }
  cluster_6 <- c(cluster_6_ID, cluster_6_dist, cluster_6_covid)
}

# A df with all the clusters nearest IDs
nearest_ids <- as.data.frame(rbind(cluster_1, cluster_2, cluster_3, cluster_4, cluster_5, cluster_6))

colnames(nearest_ids) <- c('ID', 'Distance', 'Covid')
nearest_ids$Covid <- as.factor(nearest_ids$Covid)

levels(nearest_ids$Covid)[levels(nearest_ids$Covid)== '0'] <- 'Covid Negative'
levels(nearest_ids$Covid)[levels(nearest_ids$Covid)== '1'] <- 'Covid Positive'

#extracting the nearest IDs from original PCA data

#Some Indexes are the value +250 depending on whether the subject had covid or not
nearest_values <- pca_tuned_three[c(182,61+250,150,105+250,227+250,144+250),]
library(data.table)

cluster_centres <- transpose(as.data.frame(cluster_centres))
colnames(cluster_centres) <- c('PC1', 'PC2', 'PC3')
#Adding the cluster number as a column
cluster <- c(rep(1:6, 2))
all_points <- rbind(nearest_values, cluster_centres)
#adding a col for whether its a cluster centroid or subject ID
centroid_subject <- c(rep(0,6), rep(1,6))
all_points <- cbind(all_points, as.factor(cluster), as.factor(centroid_subject))

#Adding a column for the subject ID
all_points$subjectID <- c('182', '61', '150', '105', '227', '144', rep('', 6))

colnames(all_points) <- c('PC1', 'PC2', 'PC3', 'Cluster', 'Centroid', 'Subject ID')

levels(all_points$Covid)[levels(all_points$Covid)== '0'] <- 'Covid Negative'
levels(all_points$Covid)[levels(all_points$Covid)== '1'] <- 'Covid Positive'

levels(all_points$Centroid)[levels(all_points$Centroid)== '0'] <- 'Subject ID'
levels(all_points$Centroid)[levels(all_points$Centroid)== '1'] <- 'Centroid'

#Plotting the centroids and there corresponding nearest IDs 
ggplot(data = all_points, aes(x=PC1, y=PC2,shape=Cluster, colour= Centroid)) +geom_point(size = 5) +
  labs(title = 'Each cluster centroid and its nearest subject ID') +
  geom_text(aes(label=`Subject ID`), hjust = -0.1, vjust = -0.7)

#The simpler way to find the nearest points - use the z option from the code

closest <- gmm_model[["z"]]
which.max(closest[,1])
which.max(closest[,2])
which.max(closest[,3])
which.max(closest[,4])
which.max(closest[,5])
which.max(closest[,6])

max(closest[,1])
max(closest[,2])
max(closest[,3])
max(closest[,4])
max(closest[,5])
max(closest[,6])






