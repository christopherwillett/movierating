#Testing on Validation. In this case the edx set will be the training set
#and the validation the test set.

#The previous mu was the mean for the training set portion of edx. We need it
#for all of edx.
mu_edx <- mean(edx$rating)

#Compute the regularized movie means and user effects using the penalty
#coefficient found in training.
val_reg_movie_mean <- edx %>% group_by(movieId) %>% 
  summarize(reg_mm = 1/(4.9+n())*sum(rating-mu_edx))

val_reg_user_effect <- edx %>% left_join(reg_movie_mean,by="movieId") %>% 
  group_by(userId) %>%
  summarize(uf = 1/(4.9+n())*sum(rating-mu_edx-reg_mm))

#Predict the ratings on the validation set.
val_predict <- validation %>% left_join(val_reg_movie_mean,by="movieId") %>%
  left_join(val_reg_user_effect,by = "userId") %>%
  mutate(prediction=mu_edx+reg_mm+uf) %>% pull(prediction)

#Compute the RMSE for the validation predictions.
RMSE(validation$rating,val_predict)