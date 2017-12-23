setwd("H:/Data Science/Kaggle/Spooky Author/")
library(tidytext)
library(dplyr)
library(quanteda)
library(tm)
library(tibble)

train <- read.csv("train.csv",header = TRUE, stringsAsFactors = FALSE)
train$text <- as.character(train$text)
set.seed(1234)
train <- dplyr::sample_frac(train, 0.20, replace = FALSE)

test <- read.csv("test.csv", stringsAsFactors = FALSE)


train_dtm <- train %>%
        unnest_tokens(word, text)%>%
        mutate(word = gsub("[[:punct:]]|[[:digit:]]", "", word)) %>%
        #mutate(word = gsub("  "," ",word)) %>%
        mutate(word = removeWords(x = word,words = stopwords("en"))) %>%
        mutate(word = stemDocument(word)) %>%
        count(author, id, word) %>%
        ungroup() %>%
        cast_dtm(document = id, term = word, value = n)

train_dtm <- removeSparseTerms(train_dtm,0.997)

train_df <- train_dtm %>%
        as.matrix() %>%
        as.data.frame() %>%
        rownames_to_column("id") %>%
        inner_join(y = train %>% select(id,author_name = author),by = "id") %>%
        mutate(author_name = factor(author_name))

# train_df <- train_df %>% select(-id) %>% mutate(author_name = factor(author_name))

colnames(train_df) <- paste0(colnames(train_df), "_word")
train_df <- rename(train_df, author_name = author_name_word)

library(randomForest)
set.seed(1234)
model_rf <- randomForest(author_name ~.,
                         train_df %>% select(-id_word),
                         ntree = 500,
                         mtry = 30,
                         proximity = TRUE,
                         importance = TRUE
)

# library(e1071)
# set.seed(1234)
# naivemodel <- naiveBayes(author_name ~., data = train_df %>% select(-id_word),laplace = 1)

# author <- predict(naivemodel,newdata = test_df,type = "raw")

model_rf$confusion
varImpPlot(model_rf)

test_df <- test %>%
        unnest_tokens(word, text) %>%
        mutate(word = gsub("[[:punct:]]|[[:digit:]]", "", word)) %>%
        mutate(word = removeWords(x = word, words = stopwords("en"))) %>%
        mutate(word = stemDocument(word)) %>%
        count(id, word)  %>%
        ungroup() %>%
        # filter(word %in% words_in_train) #%>%
        mutate(word = paste0(word, "_word")) %>%
        cast_dtm(id, word, n) %>%
        as.matrix() %>%
        as.data.frame() %>%
        rownames_to_column("id")

names(test_df)[2] <- "V1_word"

author <- predict(model_rf,newdata = test_df, type = 'prob')
submission <- data.frame(id = test_df$id, author)
write.csv(submission, "tmsubmission.csv",row.names = FALSE)
