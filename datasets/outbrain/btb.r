library(dplyr)
library(data.table)
library(sgd)

# Get the the probabiity of an ad click with a regularization based on impressions
getClickProb <- function(x, theta, gamma=1) {
  clicks <- x %>%
    group_by(ad_id) %>%
    summarise(
      click_prob = sum(clicked) / (n() * (theta + log(n() + gamma)))
    )
  clicks
}

fillDtNa = function(DT) {
    for (j in seq_len(ncol(DT)))
    set(DT,which(is.na(DT[[j]])),j,0)
}

clicks.train <- fread("../inputs/clicks_train.csv", sep = ",")
ad.probs <- getClickProb(clicks.train, theta = 25000)
clicks.train <- merge(clicks.train, ad.probs, by="ad_id", all.x = TRUE)
fillDtNa(clicks.train)

# train a model based on the impressions and probability of clicks
#TODO: Create/use a model that can be updated when new data is seen, then we
# can read the file a chunk at a time and gradually train the model.


#TODO: Add function to evaluate the validation set. Accuracy is not the evaluation metric
# for this competition
# test <- clicks.train[, .(max_prob=max(click_prob), clicked, click_prob), by=display_id]
# test <- test[click_prob == max_prob, ]
# sum(test$clicked) / length(test$clicked)

rm(clicks.train)
gc()

clicks.test <- fread("../inputs/clicks_test.csv", sep = ",")
clicks.test$clicked <- 0
clicks.test <- merge(clicks.test, ad.probs, by="ad_id", all.x = TRUE)
fillDtNa(clicks.test)

write.csv(clicks.test[order(-click_prob),.(ad_id=paste(ad_id, collapse=" ")),by=display_id],"basic_submission.csv", row.names = FALSE)
