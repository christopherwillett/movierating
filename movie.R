#Author: Christopher Willett with code from HarvardX capstone course as noted below.
#Date of Current Version: July 28, 2020
#Purpose: Create a rating system for movies.

#------------------------
#Wrangle the data. This code was provided by HarvardX's capstone course in 
#data science and programming in R. Note that dataWrangle.R must be in the same 
#project directory as this script. This script will take a while to run.

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


# if using R 4.0 or later
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

#Add a year of release to the movielens data frame. This will then be included with the edx and validation
#data frames. This was not included in the original HarvardX code but is needed for a model.

movielens <- movielens %>% mutate(year = str_extract(str_extract(title,"\\(\\d{4}\\)"),"\\d+"))

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)


rm(dl, ratings, movies, test_index, temp, movielens, removed)
#----------------------------------



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
  summarize(b_i=mean(rating-mu))

#Create a plot of the movie mean effects to see how prevalent the problem is.
movie_means %>% ggplot(aes(b_i))+geom_histogram(binwidth=0.1)+xlab("Movie Mean Effects")+ylab("Count")+
  ggtitle("Histogram of Movie Mean Effects")

#The movie mean effects look approximately normal, so we compute the mean and standard deviation.
mean_mm_effect <- mean(movie_means$b_i)
sd_mm_effect <- sd(movie_means$b_i)



predict2 <- test %>% 
  left_join(movie_means, by='movieId') %>% 
  mutate(prediction = mu+b_i) %>%
pull(prediction)

rmse2 <- RMSE(test$rating,predict2)

#Add the second model to the results data frame.
rmse_results <- bind_rows(rmse_results,
                          data.frame(method="Movie Mean Model",RMSE=rmse2))

#To improve on this, we'll determine how far from the second model each user is 
# on average. We'll use that to adjust the second model to get a more 
#accurate prediction.

user_effect <- train %>% left_join(movie_means,by='movieId') %>% 
  group_by(userId) %>% summarize(b_u=mean(rating-mu-b_i))

predict3 <- test %>% left_join(movie_means,by='movieId') %>% left_join(user_effect,by='userId') %>%
  mutate(prediction=mu+b_i+b_u) %>%
  pull(prediction)

rmse3 <- RMSE(test$rating,predict3)

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
    summarize(b_i = 1/(p+n())*sum(rating-mu))
  
  reg_user_effect <- train %>% left_join(reg_movie_mean,by="movieId") %>% 
    group_by(userId) %>%
    summarize(b_u = 1/(p+n())*sum(rating-mu-b_i))
  
  
  predict4 <- test %>% left_join(reg_movie_mean,by="movieId") %>%
    left_join(reg_user_effect,by = "userId") %>%
    mutate(prediction=mu+b_i+b_u) %>% pull(prediction)
  
  return(RMSE(test$rating,predict4))
}

#The below code will take a long time to run. The end result is that the 
#penalty constant that minimizes RMSE is 4.9
#penalties <- seq(0,10,0.1)
#rmse_by_penalty <- sapply(penalties,rmse_by_penalty)

#Pick off the min and the penalty constant.
#rmse4 <- min(rmse_by_penalty)
#penalty_min <- (which.min(rmse_by_penalty)-1)*0.1

#The RMSE is minimized for a penalty coefficient of 4.9 and we use it here.
reg_movie_mean <- train %>% group_by(movieId) %>% 
  summarize(b_i = 1/(4.9+n())*sum(rating-mu))

reg_user_effect <- train %>% left_join(reg_movie_mean,by="movieId") %>% 
  group_by(userId) %>%
  summarize(b_u = 1/(4.9+n())*sum(rating-mu-b_i))

predict4 <- test %>% left_join(reg_movie_mean,by="movieId") %>%
  left_join(reg_user_effect,by = "userId") %>%
  mutate(prediction=mu+b_i+b_u) %>% pull(prediction)


rmse4 <- RMSE(test$rating,predict4)
rmse_results <- bind_rows(rmse_results,
                          data.frame(method="Regularized Movie Mean and User Effects",RMSE=rmse4))


#We can utilize the date of release as well. Movies that were released multiple decades ago and are in the 
#data set are probably regarded as good movies. No reason to keep around a bad movie. As such, it is reasonable 
#to assume that the older a movie is, the higher rated it is in general. We can explore this below. When we 
#use this in our model it will be done just on train, but for exploration we'll do it on the whole edx set.
#We'll compute the year-by-year mean and determine how far above the global mean each year is. Note that 
#a positive residual indicates that a year is above the global mean.

year_effect_explore <- edx %>% group_by(year) %>% summarise(year_mean=mean(rating),res_error=year_mean-mu)
year_effect_explore %>% ggplot(aes(year,res_error))+geom_bar(stat="Identity")+
  ggtitle("Year-by-Year Mean Residual Error")+xlab("Year")+ylab("Deviation from Global Mean")+
  theme(axis.text.x=element_text(angle=90))

#Since there is evidence of an effect from the year of release, we include this in our model. Note that 
#regularization is not used here due to the large number of movies in each year. Although it might be desireable
#to explore whether or not regularization could be helpful here, the limitations of hardware on my end 
#makes this not feasible.

year_effect <- train %>% left_join(reg_movie_mean,by="movieId") %>% left_join(reg_user_effect,by="userId") %>%
  group_by(year) %>% summarise(b_y = mean(rating-mu-b_i-b_u))

predict5 <- test %>% left_join(reg_movie_mean,by="movieId") %>% left_join(reg_user_effect,by="userId") %>%
  left_join(year_effect,by="year") %>% mutate(prediction = mu+b_i+b_u+b_y) %>%
  pull(prediction)
rmse5 <- RMSE(test$rating,predict5)

rmse_results <- bind_rows(rmse_results,
                          data.frame(method="Regularized Movie Mean and User Effects, with Year Effect",RMSE=rmse5))
#Some users show a strong preference for one genre of movie over another. For example, they might 
#prefer comedy movies and not prefer thrillers. As such, they might rate comedy movies higher and 
#thriller movies lower than the model developed so far might predict.

#Create a data set that has some basic information about the genres. This was used in the 
#report to gather some basic statistics.
genre_explore <- edx %>% group_by(genres) %>% summarise(genre_mean=mean(rating),n_genre=n()) %>% 
  arrange(desc(genre_mean))

top_5_by_mean <- genre_explore[1:5,]
bottom_5_by_mean <- genre_explore[792:797,]


#In the data exploration for the genre effect, it seems clear that a regulariation approach
#is needed. 

#This was the function used to determine the penalty term in regularisation by genre.
rmse_by_penalty_genre <- function(p){
  genre_effect <- train %>%  left_join(reg_movie_mean,by="movieId") %>% left_join(reg_user_effect,by="userId") %>%
    left_join(year_effect,by="year") %>% group_by(genres) %>% 
    summarise(b_g = 1/(n()+p)*sum(rating-mu-b_i-b_u+b_y))
  
  predict6 <- test %>% left_join(reg_movie_mean,by="movieId") %>% left_join(reg_user_effect,by="userId") %>%
    left_join(year_effect,by="year") %>% left_join(genre_effect,by="genres") %>% 
    mutate(prediction = mu+b_i+b_u+b_y+b_g) %>%
    pull(prediction)
  
  return(RMSE(test$rating,predict6))
}

#Uncomment below if you want to fiddle around with tuning the penalty term.
#penalties_genre <- seq(33,37,0.5)
#rmse_by_penalty_genre_set <- sapply(penalties_genre,rmse_by_penalty_genre)


genre_effect <- train %>%  left_join(reg_movie_mean,by="movieId") %>% left_join(reg_user_effect,by="userId") %>%
  left_join(year_effect,by="year") %>% group_by(genres) %>% 
  summarise(b_g = 1/(n()+35)*sum(rating-mu-b_i-b_u-b_y))

predict6 <- test %>% left_join(reg_movie_mean,by="movieId") %>% left_join(reg_user_effect,by="userId") %>%
  left_join(year_effect,by="year") %>% left_join(genre_effect,by="genres") %>% 
  mutate(prediction = mu+b_i+b_u+b_y+b_g) %>%
  pull(prediction)

rmse6 <- RMSE(test$rating,predict6)

rmse_results <- bind_rows(rmse_results,
                          data.frame(method="Regularized Movie Mean and User Effects, with Year and Genre Effects",RMSE=rmse6))

#-------------------------------------------------

#As an application, we can recommend 5 movies to a user using the new model.
#We can select a user and then apply the model just to them. Because 
#the regularized mean and user effect are behind a function, they have to 
#be pulled out and re-run. Changing the function to accept test and train 
#sets might be better.


#Client userId 31324. This client has at least 50 reviews.
client_Id <- 31324

#Pull their regularized user effect from the above data frame.
client_b_u <- reg_user_effect %>% filter(userId==client_Id) %>% pull(b_u)

#make a prediction for each movie just for this user. Note that a simpler model is used just 
#for the sake of shorter code.
client_pred <- reg_movie_mean %>% mutate(prediction = mu+b_i+client_b_u) %>% 
  select(-b_i) 

#Determine which movies they have reviewed.
client_reviewed <- edx %>% filter(userId==client_Id) %>% select(c(movieId))

#And which they have not.
client_not_reviewed <- setdiff(select(client_pred,-prediction),client_reviewed)

#Generate a master recommendation list for the client of movies they have 
#not seen.
client_recs <- client_pred %>% filter(movieId %in% client_not_reviewed$movieId) %>%
  arrange(desc(prediction))

#Report out the top five movies that the model recommends for them that 
#they have not yet reviewed.

top5 <- edx %>% group_by(movieId) %>% 
  filter(movieId %in% client_recs[1:5,1]$movieId) %>% select(movieId,title) %>% unique()

#----------------------------------

#Determine the effectiveness of the model on the validation set. 
#In this case the edx set will be the training set
#and the validation the test set.

#The previous mu was the mean for the training set portion of edx. We need it
#for all of edx.
mu_edx <- mean(edx$rating)

#Compute the regularized movie means and user effects using the penalty
#coefficient found in training.
val_reg_movie_mean <- edx %>% group_by(movieId) %>% 
  summarize(b_i = 1/(4.9+n())*sum(rating-mu_edx))

val_reg_user_effect <- edx %>% left_join(val_reg_movie_mean,by="movieId") %>% 
  group_by(userId) %>%
  summarize(b_u = 1/(4.9+n())*sum(rating-mu_edx-b_i))

val_year_effect <- edx %>% left_join(val_reg_movie_mean,by="movieId") %>% left_join(val_reg_user_effect,by="userId") %>%
  group_by(year) %>% summarise(b_y = mean(rating-mu_edx-b_i-b_u))

val_genre_effect <- edx %>%  left_join(val_reg_movie_mean,by="movieId") %>% 
  left_join(val_reg_user_effect,by="userId") %>%
  left_join(val_year_effect,by="year") %>% group_by(genres) %>% 
  summarise(b_g = 1/(n()+35)*sum(rating-mu_edx-b_i-b_u-b_y))


#Predict the ratings on the validation set.
val_predict <- validation %>% left_join(val_reg_movie_mean,by="movieId") %>%
  left_join(val_reg_user_effect,by = "userId") %>%
  left_join(val_year_effect,by="year") %>%
  left_join(val_genre_effect,by="genres") %>%
  mutate(prediction=mu_edx+b_i+b_u+b_y+b_g) %>% pull(prediction)

#Compute the RMSE for the validation predictions.
val_rmse <- RMSE(validation$rating,val_predict)
