# =========================== Import Packages ================================ #
install.packages('TSA')
install.packages('tseries')
install.packages('forecast')
install.packages("lmtest")

library(TSA)
library(tseries)
library(psych)
library(ggplot2)
library(forecast)
library(lmtest)
library(zoo)

# ========================= II. Data Prepossessing =========================== #
# Import Data
data <- read.csv("BTC-USD.csv")
head(data)

# Check for any missing values
sum(is.na(data))

# Basic Plot for all variables against time
plot(data, main='Time Series of Daily Bitcoin Price')

# Extract the "Close" variable
close <- data[,"Close"]
head(close)

# Plot: "Close" against time
plot(close, type="l", main='Daily Bitcoin Price', ylab="Close")

# Normal QQ-Plot
qqnorm(close)
qqline(close, col=2)

# ======================= III. Linear Trend Model ============================ #
time <- 1:length(close)
linear_model <- lm(close ~ time)
summary(linear_model)

Q1 <- quantile(close, 0.25)
Q3 <- quantile(close, 0.75)
IQR = Q3-Q1
Upperbound_outlier <- Q3 + 1.5*IQR
Lowerbound_outlier <- Q1 - 1.5*IQR

outlier_proportion <- (sum(close>Upperbound_outlier)+sum(close<Lowerbound_outlier))/length(close)

# Visualization
plot(close, type="l", main='Daily Bitcoin Price', ylab="Close")
abline(linear_model, col="red")

# Check whether the residuals ~ N
linear_residuals <- linear_model$residuals
qqnorm(linear_residuals)
qqline(linear_residuals, col="red")

# Calculate RMSE
linear_rmse <- sqrt(mean(linear_residuals^2))
print(paste("Root Mean Squared Error (RMSE): ", linear_rmse))

# Calculate MAE
linear_mae <- mean(abs(linear_residuals))
print(paste("Mean Absolute Error (MAE): ", linear_mae))

# ========================== IV. Quadratic Model ============================= #
time_squared <- time^2
quadratic_model <- lm(close ~ time + time_squared)
summary(quadratic_model)

# Plotting the time series and quadratic trend model   
plot(time, close, type = "l", xlab = "Time", ylab = "BTC Price", main = "Quadratic Trend Model of BTC price")
lines(time, predict(quadratic_model), col = "red")
legend("topleft", legend = c("Actual Price", "Quadratic Model"), col = c("black", "red"), lty = c(1, 1), cex=0.75)

# Calculate RMSE
print(paste("Root Mean Squared Error (RMSE): ", sqrt(mean(quadratic_model$residuals^2))))

# Calculate MAE
print(paste("Mean Absolute Error (MAE): ", mean(abs(quadratic_model$residuals))))

# ========================== V. Moving Average= ============================== #
# Calculate the moving average with a window size of 30 
moving_avg <- ma(close, order = 30)

# Plot the original time series and the moving average
plot(close, main = "Time Series with Moving Average")
lines(moving_avg, col = "red")
legend("topleft", legend = c("Time Series", "Moving Average"), col = c("black", "red"), lty = c(1, 1))

# Calculate RMSE
ma_rmse <- sqrt(mean((close-moving_avg)^2, na.rm = TRUE))
print(paste("Root Mean Squared Error (RMSE): ", ma_rmse))

# Calculate MAE
ma_mae <- mean(abs(close-moving_avg), na.rm = TRUE)
print(paste("Mean Absolute Error (MAE): ", ma_mae))

# ======================= VI. Centered Moving Average ======================== #
centered_ma <- rollmean(close, k = 30, align = "center", fill = NA)

# Plot the original time series and the centered moving average
plot(close, main = "BTC Prices with Centered Moving Average")
lines(centered_ma, col = "red")
legend("topleft", legend = c("BTC Price", "Centered Moving Average"), col = c("black", "red"), lty = c(1, 1))

# Calculate RMSE
centered_ma_rmse <- sqrt(mean((close - centered_ma)^2, na.rm = TRUE))
print(paste("Root Mean Squared Error (RMSE): ", centered_ma_rmse))

# Calculate MAE
centered_ma_mae <- mean(abs(close - centered_ma), na.rm = TRUE)
print(paste("Mean Absolute Error (MAE): ", centered_ma_mae))

# ======================= VII. Exponential Smoothing ========================= #
ets_BTC<- ets(close)

# Plot
plot(close, main = "BTC Prices with Exponential Smoothing", xlab = "Date", ylab = "BTC price")
lines(ets_BTC$fitted, col = "red", lwd = 2)

# Calculate RMSE
ets_rmse <- sqrt(mean((close - ets_BTC$fitted)^2, na.rm = TRUE))
print(paste("Root Mean Squared Error (RMSE): ", ets_rmse))

# Calculate MAE
ets_mae <- mean(abs(close - ets_BTC$fitted), na.rm = TRUE)
print(paste("Mean Absolute Error (MAE): ", ets_mae))

# ===================== VIII. ARIMA Model (Final) ============================ #
# Different ACF & PACF Plots
par(mfrow=c(1,2))
acf(close,lag.max = 800, main="ACF plot of Bitcoin Value")
pacf(close,lag.max = 800, main="PACF plot of Bitcoin Value")

par(mfrow=c(1,1))
pacf(close,lag.max = 10, main="PACF plot of Bitcoin Value")

# ADF test for stationary data
adf.test(close, k=2)

close.diff1 <- diff(close,1)
ts.plot(close.diff1)

adf.test(close.diff1, k=2)

# ACF & PACF Plots with 1st Differenced Bitcoin Value
par(mfrow=c(1,2))
acf(close.diff1, lag.max = 1800, main = "ACF plot of 1st Differenced Bitcoin Value")
pacf(close.diff1, lag.max = 1800, main = "PACF plot of 1st Differenced Bitcoin Value") 

# ARIMA Model Selection
auto.arima(close,stepwise = F,approximation = F,trace = T)
arima_model = Arima(close, order = c(2,1,2), include.drift = FALSE)
checkresiduals(arima_model)

# ARIMA Model Forecast Value & Plots 
(arima_model_forecast = forecast(arima_model,h = 27))

par(mfrow=c(1,2))
plot(arima_model_forecast, xlim=c(1700,1900), main = "ARIMA Forecast")
plot(close,type="l", xlim=c(1700,1900), main = "Realised Values")

# Calculating RMSE
arima_model_forecast_mean <- arima_model_forecast$mean
arima_rmse <- sqrt(mean((arima_model_forecast_mean - close[1801:1827])^2))
print(paste("Root Mean Squared Error (RMSE): ", arima_rmse))

# Calculating MSE
arima_mse <- mean((arima_model_forecast_mean - close[1801:1827])^2)
print(paste("Mean Squared Error (MSE): ", arima_mse))

# Calculating MAE
arima_mae <- mean(abs(arima_model_forecast_mean - close[1801:1827]))
print(paste("Mean Absolute Error (MAE): ", arima_mae))

# ======================== IX Forecast Comparsion ============================ #
# Combined (All) Plot
ts_of_forecast_error_of_linear_model <- ts(linear_residuals)
ts_of_forecast_error_of_quadratic_model <- ts(quadratic_model$residuals)
ts_of_forecast_error_of_moving_avg <- ts(close-moving_avg)
ts_of_forecast_error_of_centered_moving_avg <- ts(close - centered_ma)
ts_of_forecast_error_of_ets <- ts(close - ets_BTC$fitted)

plot(ts_of_forecast_error_of_ets, type = "l", col = "blue", xlab = "Date", ylab = "Value", main = "Forecast Error of Different Model", ylim=c(min(ts_of_forecast_error_of_linear_model),max(ts_of_forecast_error_of_linear_model)))
lines(ts_of_forecast_error_of_moving_avg, type = "l", col = "black", lwd=2)
lines(ts_of_forecast_error_of_centered_moving_avg, type = "l", col = "red")
lines(ts_of_forecast_error_of_linear_model, type = "l", col = "green")
lines(ts_of_forecast_error_of_quadratic_model, type = "l", col = "purple")

par(mfrow=c(1,1))
par(xpd = TRUE)
legend("topleft", legend = c("Linear", "Quadratic","Moving Avg","Centered Moving Avg","Exponential Smoothing"), col = c("green", "purple","black","red","blue"), lty = c(1,1,1,1,1),bty = "n", xjust = 0, yjust = 0)
par(xpd = FALSE)

# Combined (Smoothing) Plot
plot(ts_of_forecast_error_of_ets, type = "l", col = "blue", xlab = "Date", ylab = "Value", main = "Forecast Error of Different Smoothing Method", ylim=c(min(ts_of_forecast_error_of_moving_avg,na.rm=TRUE),max(ts_of_forecast_error_of_moving_avg,na.rm=TRUE)))

lines(ts_of_forecast_error_of_moving_avg, type = "l", col = "black", lwd=2)
lines(ts_of_forecast_error_of_centered_moving_avg, type = "l", col = "red")

par(xpd = TRUE)
legend("topleft", legend = c("Moving Avg","Centered Moving Avg","Exponential Smoothing"), col = c("black","red","blue"), lty = c(1,1,1),bty = "n", xjust = 0, yjust = 0, cex=0.9)
par(xpd = FALSE)