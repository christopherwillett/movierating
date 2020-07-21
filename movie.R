#R Script to predict user ratings of movies.

#Wrangle the data. This code was provided by HarvardX's capstone course in 
#data science and programming in R. Note that dataWrangle.R must be in the same 
#project directory as this script. This script will take a while to run.
source("dataWrangle.R")

#The data frames edx and validation hold the movie review data. edx will be used
#for development of the algorithm and validation will be the final test set.
#We'll split edx into a train and test set instead of using cross validation
#as the data set is quite large.
set.seed(1974,sample.kind="Rounding")
test_ind <- createDataPartition(y=edx$rating,times=1,p=0.2,list=FALSE)
test <- edx[test_ind,]
train <- edx[-test_ind,]

#It might be the case that our test set does include all the movies or users
#from the train set, and conversely. So, we fix this at the start.

test <- test %>% semi_join(train,by='movieId') %>% semi_join(train,by='userId')

#Metric for success is to minimize the root mean squared error of our predictions.
RMSE <- function(actual,predict){
  sqrt(mean((actual-predict)^2))
} 

#Model 1: Just use the global mean of ratings for our prediction. This assumes
#that every user would just give the global mean for their prediction.
mu <- mean(train$rating)
rmse1 <- RMSE(test$rating,mu)

#This results in an RMSE of about 1.06, which is a pretty big gap. Several models
#will be developed and a data frame will be used to store the results.

rmse_results <- data.frame(method="Global Mean",RMSE = rmse1)

#Model 2: Account for the ratings of individual movies. Movies that are 
#consistently highly rated are penalized under model 1, and movies that are
#consistently poorly rated are helped under model 1. In this case we will
#add an effect to our model to account for this. To do this, we find the mean
#of how far from the global mean a given rating is. For each user, we'll 
#predict the global mean plus this effect. 

movie_means <- train %>% group_by(movieId) %>% 
  summarize(movie_mean_effect=mean(rating-mu))

model2 <- mu + (test %>% 
  left_join(movie_means, by='movieId') %>%
  pull(movie_mean_effect))

rmse2 <- RMSE(test$rating,model2)

#Add the second model to the results data frame.
rmse_results <- bind_rows(rmse_results,
                          data.frame(method="Movie Mean Model",RMSE=rmse2))

#To improve on this, we'll determine how far from the second model each user is 
# on average. We'll use that to adjust the second model to get a more 
#accurate prediction.

user_effect <- train %>% left_join(movie_means,by='movieId') %>% 
  group_by(userId) %>% summarize(uf=mean(rating-mu-movie_mean_effect))

model3 <- test %>% left_join(movie_means,by='movieId') %>% left_join(user_effect,by='userId') %>%
  mutate(prediction=mu+movie_mean_effect+uf) %>%
  pull(prediction)

rmse3 <- RMSE(test$rating,model3)

rmse_results <- bind_rows(rmse_results,
                          data.frame(method="Movie Mean with User Effect Model",RMSE=rmse3))

#We can attempt to regularize the ratings by introducing a penalty term that
#effectively penalizes movies with few ratings. The penalty term is 
#proportional to the sum of the squares of the mean movie effects.
#We regularize both the mean movie rating and the user effect.
#We can tune the proportionality constant to minimize RMSE. The code for doing so is below, but is 
#commented out as it will take a while to run. The result of the minimization
#is used below.

rmse_by_penalty <- function(p){
  reg_movie_mean <- train %>% group_by(movieId) %>% 
    summarize(reg_mm = 1/(p+n())*sum(rating-mu))
  
  reg_user_effect <- train %>% left_join(reg_movie_mean,by="movieId") %>% 
    group_by(userId) %>%
    summarize(reg_user_effect = 1/(p+n())*sum(rating-mu-reg_mm))
  
  
  model4 <- test %>% left_join(reg_movie_mean,by="movieId") %>%
    left_join(reg_user_effect,by = "userId") %>%
    mutate(prediction=mu+reg_mm+reg_user_effect) %>% pull(prediction)
  
  return(RMSE(test$rating,model4))
}

#The below code will take a long time to run. The end result is that the 
#penalty constant that minimizes RMSE is 4.9
#penalties <- seq(0,10,0.1)
#rmse_by_penalty <- sapply(penalties,find_penalty)

#Pick off the min and the penalty constant.
#rmse4 <- min(rmse_by_penalty)
#penalty_min <- (which.min(rmse_by_penalty)-1)*0.1

#The RMSE is minimized for a penalty coefficient of 4.9 and we use it here.
reg_movie_mean <- edx %>% group_by(movieId) %>% 
  summarize(reg_mm = 1/(4.9+n())*sum(rating-mu))

reg_user_effect <- edx %>% left_join(reg_movie_mean,by="movieId") %>% 
  group_by(userId) %>%
  summarize(reg_user_effect = 1/(4.9+n())*sum(rating-mu-reg_mm))

model4 <- test %>% left_join(reg_movie_mean,by="movieId") %>%
  left_join(reg_user_effect,by = "userId") %>%
  mutate(prediction=mu+reg_mm+reg_user_effect) %>% pull(prediction)


rmse4 <- RMSE(test$rating,model4)
rmse_results <- bind_rows(rmse_results,
                          data.frame(method="Regularized Movie Mean and User Effects",RMSE=rmse4))

#As an application, we can recommend 5 movies to a user using the new model.
#We can select a user and then apply the model just to them. Because 
#the regularized mean and user effect are behind a function, they have to 
#be pulled out and re-run. Changing the function to accept test and train 
#sets might be better.


#Client userId 31324. This client has at least 50 reviews.
client_Id <- 31324

#Pull their regularized user effect from the above data frame.
client_ruf <- reg_user_effect %>% filter(userId==client_Id) %>% pull(reg_user_effect)

#make a prediction for each movie just for this user.
client_pred <- reg_movie_mean %>% mutate(prediction = mu+reg_mm+client_ruf) %>% 
  select(-reg_mm) 

#Determine which movies they have reviewed.
client_reviewed <- edx %>% filter(userId==client_Id) %>% select(c(movieId))

#And which they have not.
client_not_reviewed <- setdiff(select(client_pred,-prediction),client_reviewed)

#Great a master recommendation list for the client of movies they have 
#not seen.
client_recs <- client_pred %>% filter(movieId %in% client_not_reviewed$movieId) %>%
  arrange(desc(prediction))

#Report out the top five movies that the model recommends for them that 
#they have not yet reviewed.

top5 <- edx %>% group_by(movieId) %>% 
  filter(movieId %in% client_recs[1:5,1]$movieId) %>% select(movieId,title) %>% unique()


