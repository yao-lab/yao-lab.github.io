install.packages("lunar")
install.packages("RcppRoll")
# install.packages("timetk")
install.packages("tidymodels")
install.packages("tidyquant")
install.packages("modeltime")
library(tidyverse)
library(data.table)
library(RcppRoll) 
library(timetk)
library(tidymodels)
library(tidyquant)
library(modeltime)
library(lubridate)
library(lunar)

print(getwd())

# Data
main <- "C:/Users/Liu Yi/m5-forecasting"

calendar <- fread(file.path(main, "calendar.csv"), stringsAsFactors = TRUE, 
                  drop = c("wday", "month", "year", "weekday"))

data <- fread(file.path(main, "sales_train_validation.csv"), stringsAsFactors = TRUE)
prices <- fread(file.path(main, "sell_prices.csv"), stringsAsFactors = TRUE)

# Functions
d2int <- function(X) {
  X %>% extract(d, into = "d", "([0-9]+)", convert = TRUE)
}

FIRST <- 1914 # Start to predict from this "d"
LENGTH <- 28  # Predict so many days

data[, paste0("d_", FIRST:(FIRST + LENGTH - 1))] <- NA

data <- data %>% 
  mutate(id = gsub("_validation", "", id)) %>%
  gather("d", "demand", -id, -item_id, -dept_id, -cat_id, -store_id, -state_id) %>% 
  d2int() %>% 
  left_join(calendar %>% d2int(), by = "d") %>% 
  inner_join(prices, by = c("store_id", "item_id", "wm_yr_wk")) %>% 
  select(-wm_yr_wk, -item_id, -dept_id, -store_id, -d) %>% 
  mutate(demand = as.numeric(demand), date = as.Date(date))


head(data)

# write.csv(data,"M5_sales.csv", row.names = FALSE)

## feature engineering 

best_ids <- read_csv('C:/Users/Liu Yi/m5-forecasting/ids.csv')

data <- data %>% 
  filter(id %in% best_ids$id)

# One hot encode events
data <- data %>% 
  mutate(isEvent = as.numeric(as.logical(nchar(as.character(data$event_name_1))))) %>%
  mutate(snap_CA = as.numeric(snap_CA),snap_TX = as.numeric(snap_TX),snap_WI = as.numeric(snap_WI)) %>%
  select(-event_name_1, -event_name_2)

# Price Momentum - Weekly
data <- data %>% 
  group_by(id) %>% 
  mutate(wk.lag = dplyr::lag(sell_price, n=7, default = NA)) %>% 
  mutate(price_momentum_wk = sell_price - wk.lag) %>%
  ungroup()

# Create features from date
# Add year, month, day of month, day of week and whether it is a weekend (dayofweek is either 7 or 1), and the moonphase
# Note that Sunday is denoted by 1 
data <- data %>% mutate(
  year = year(date), 
  month = month(date), 
  day = mday(date), 
  dayofweek = wday(date), 
  moonphase = lunar.phase(date, name=TRUE)) %>%
  mutate(iswknd = ifelse(dayofweek == 7 | dayofweek == 1, 1, 0)) # Add Weekend

head(data)

# Generate lags 
lag_transformer <- function(data) { 
  interval <- c(7, 14, 30, 60, 180)
  data %>% 
    group_by(id) %>% 
    tk_augment_lags(demand, .lags = interval, .names = str_c("Lag_", interval)) %>% # lags
    tk_augment_slidify(.value = demand, .period = interval, .f = AVERAGE, 
                       .partial = TRUE, .names = str_c("MA_", interval)) %>% # rolling_mean
    tk_augment_slidify(.value = demand, .period = interval, .f = STDEV, 
                       .partial = TRUE, .names = str_c("MSD_", interval)) %>% # rolling_sd
    ungroup() 
}

data <- recipe(demand ~ ., data = data) %>%
  step_dummy(contains("event_type")) %>%
  step_dummy(moonphase, cat_id, state_id) %>%
  step_zv(all_predictors()) %>%
  prep() %>%
  juice() %>%
  lag_transformer()


head(data)

nrow(unique(data[rowSums(is.na(data)) > 0, 'date'])) == 28

