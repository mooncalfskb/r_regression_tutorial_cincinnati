#http://uc-r.github.io/logistic_regression
#rm(list = ls())
# install.packages("tidyverse")
# install.packages("modelr")
# install.packages("broom")
#install.packages("ISLR")
#install.packages("caret")
#install.packages('pscl')
# Packages
library(tidyverse)  # data manipulation and visualization
library(modelr)     # provides easy pipeline modeling functions
library(broom)      # helps to tidy up model outputs
library(caret)
library(pscl)

# Load data 
(default <- as_tibble(ISLR::Default))
# A tibble: 10,000 x 4
# default student balance income
# <fct>   <fct>     <dbl>  <dbl>
# 1 No      No       730. 44362.
# 2 No      Yes      817. 12106.
# 3 No      No      1074. 31767.
# 4 No      No       529. 35704.

#set seed and split data 60/40
set.seed(123)
sample <- sample(c(TRUE, FALSE), nrow(default), replace = T, prob = c(0.6,0.4))
train <- default[sample, ]
test <- default[!sample, ]

#create the glm model. Note that family = "binomial" tells it to use logistic regression
model1 <- glm(default ~ balance, family = "binomial", data = train)

summary(model1)
#shows that balance is statistically significant

# Call:
#   glm(formula = default ~ balance, family = "binomial", data = train)
# 
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -2.2905  -0.1395  -0.0528  -0.0189   3.3346  
# 
# Coefficients:
#   Estimate        Std.          Error z value Pr(>|z|)    
# (Intercept)   -1.101e+01  4.887e-01  -22.52   <2e-16 ***
#   balance      5.669e-03  2.949e-04   19.22   <2e-16 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# IMPORTANT: The null deviance represents the difference between a model 
#with only the intercept (which means “no predictors”) and a saturated model 
#(a model with a theoretically perfect fit). 
#The goal is for the model deviance (noted as Residual deviance) to be lower; 
#smaller values indicate better fit. In this respect, the null model provides 
#a baseline upon which to compare predictor models.

# Null deviance: 1723.03  on 6046  degrees of freedom
# Residual deviance:  908.69  on 6045  degrees of freedom
# AIC: 912.69
# 
# Number of Fisher Scoring iterations: 8

default %>%
  mutate(prob = ifelse(default == "Yes", 1, 0)) %>%
  ggplot(aes(balance, prob)) +
  geom_point(alpha = .15) +
  geom_smooth(method = "glm", method.args = list(family = "binomial")) +
  ggtitle("Logistic regression model fit") +
  xlab("Balance") +
  ylab("Probability of Default")

#use this command to check fitness of model.
tidy(model1)
##          term      estimate   std.error statistic       p.value
## 1 (Intercept) -11.006277528 0.488739437 -22.51972 2.660162e-112
## 2     balance   0.005668817 0.000294946  19.21985  2.525157e-82

#Bear in mind that the coefficient estimates from logistic regression characterize 
#the relationship between the predictor and response variable on a log-odds scale 
#(see Ch. 3 of ISLR1 for more details). Thus, we see that 
#^β1 = 0.0057
#this indicates that an increase in balance is associated with an increase 
#in the probability of default. To be precise, a one-unit increase in balance 
#is associated with an increase in the log odds of default by 0.0057 units.

#We can further interpret the balance coefficient as - for every one dollar 
#increase in monthly balance carried, the odds of the customer defaulting 
#increases by a factor of 1.0057.

exp(coef(model1))
#(Intercept)       balance 
#1.659718e-05 1.005685e+00 

confint(model1)
##                     2.5 %        97.5 %
## (Intercept) -12.007610373 -10.089360652
## balance       0.005111835   0.006269411
#I think this above means thta the Intercept (no predictors) has a confidence of -12, 
#and -10, but balance has extremely small differences.

#note the prediction comes from this:
#B0 appears to be the estimate of the Intercept (which I guess is with no predictors?) 
#and B1 = the estimate of the balance
#tidy(model1)
##          term      estimate   std.error statistic       p.value
## 1 (Intercept) -11.006277528 0.488739437 -22.51972 2.660162e-112
## 2     balance   0.005668817 0.000294946  19.21985  2.525157e-82

#e −11.0063  + (0.0057 × 1000)
#__________________________________
#1 + e −11.0063  + (0.0057 × 1000)

predict(model1, data.frame(balance = c(1000, 2000)), type = "response")
##           1           2 
## 0.004785057 0.582089269

## different regression, using value of "student" in the dataset

model2 <- glm(default ~ student, family = "binomial", data = train)

tidy(model2)
# A tibble: 2 x 5
# term        estimate std.error statistic p.value
# <chr>           <dbl>     <dbl>      <dbl>   <dbl>
# 1 (Intercept)   -3.55     0.0934    -38.1      0      
# 2 studentYes     0.441    0.149     2.96       0.00311

model3 <- glm(default ~ balance + income + student, family = "binomial", data = train)
tidy(model3)
#However, the coefficient for the studentYes variable is negative, 
#indicating that students are less likely to default than non-students.
#term            estimate std.error statistic  p.value
# 1 (Intercept) -10.9        0.648       -16.8   1.47e-63
# 2 balance       0.00591    0.000310     19.0   7.90e-81
# 3 income       -0.00000501 0.0000108    -0.465 6.42e- 1
# 4 studentYes   -0.809      0.313        -2.58  9.78e- 3

#student with a credit card balance of $1,500 and an income of $40,000
#https://www.symbolab.com/solver/exponents-multiplication-calculator/%5Cleft(2.71828%5Cright)%5E%7B18.9626%7D
#e = 2.71828
#B0 = -10.907
#B1 (balance coeffi) X balance
#B2 (income coef) x income
#B3 (student coef) x student = 1
#2.71828 to the ( −10.907 + (0.00591 × 1500) + (− 0.00001 × 40) + (− 0.809 × 1)
#I did this in excel. See the math_worksheet. It came out right.

caret::varImp(model3)
##               Overall
## balance    19.0403764
## income      0.4647343
## studentYes  2.5835947

anova(model1, model3, test = "Chisq")

# Analysis of Deviance Table
# 
# Model 1: default ~ balance
# Model 2: default ~ balance + income + student
# Resid. Df Resid. Dev Df Deviance Pr(>Chi)   
# 1      6045     908.69                        
# 2      6043     895.02  2   13.668 0.001076 **
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

#The results indicate that, compared to model1, 
#model3 reduces the residual deviance by over 13 
#(remember, a goal of logistic regression is to find a model 
#that minimizes deviance residuals). 

#use mcFadden test to see fitness of model.
list(model1 = pscl::pR2(model1)["McFadden"],
     model2 = pscl::pR2(model2)["McFadden"],
     model3 = pscl::pR2(model3)["McFadden"])


#note that model with mcFadden above .40 is good. 
#note that model 3 is slightly better than model1
# $model1
# McFadden 
# 0.4726215 
# 
# $model2
# McFadden 
# 0.004898314 
# 
# $model3
# McFadden 
# 0.4805543 


### Another test: 3 deviations.
#use augment to do this.
model1_data <- augment(model1) %>% 
  mutate(index = 1:n())

ggplot(model1_data, aes(index, .std.resid, color = default)) + 
  geom_point(alpha = .5) +
  geom_ref_line(h = 3)

#look for people where std.resid > 3. weird outliers. see chart.
model1_data %>% 
  filter(abs(.std.resid) > 3)

#what the fuck? people with low balance that defaulted anyway.
#
# A tibble: 8 x 10
# default balance .fitted .se.fit .resid     .hat .sigma .cooksd .std.resid index
# <fct>     <dbl>   <dbl>   <dbl>  <dbl>    <dbl>  <dbl>   <dbl>      <dbl> <int>
# 1 Yes       1119.   -4.66   0.175   3.06 0.000284  0.386  0.0151       3.06   271
# 2 Yes       1119.   -4.66   0.175   3.06 0.000284  0.386  0.0151       3.06   272
# 3 Yes       1135.   -4.57   0.171   3.03 0.000297  0.386  0.0144       3.03  1253
# 4 Yes       1067.   -4.96   0.189   3.15 0.000246  0.386  0.0175       3.15  1542
# 5 Yes        961.   -5.56   0.216   3.33 0.000180  0.385  0.0232       3.33  3488
# 6 Yes       1144.   -4.52   0.169   3.01 0.000303  0.386  0.0140       3.01  4142
# 7 Yes       1013.   -5.26   0.203   3.25 0.000211  0.385  0.0203       3.25  5058
# 8 Yes        962.   -5.55   0.216   3.33 0.000180  0.385  0.0232       3.33  5709
# > 

#Similar to linear regression we can also identify influential observations with Cook’s distance values. 
#Here we identify the top 5 largest values.
##Another test. Cooks distance. Model 1, default compared to balance.
##outliers 
plot(model1, which = 4, id.n = 5)

#And we can investigate these further as well. 
#Here we see that the top five influential points include:
#those customers who defaulted with very low balances and
#two customers who did not default, yet had balances over $2,000

model1_data %>% 
  top_n(5, .cooksd)

# default balance .fitted .se.fit .resid     .hat .sigma .cooksd .std.resid index
# <fct>     <dbl>   <dbl>   <dbl>  <dbl>    <dbl>  <dbl>   <dbl>      <dbl> <int>
#   1 No        2388.    2.53   0.241  -2.28 0.00398   0.387  0.0252      -2.29  2382
# 2 Yes        961.   -5.56   0.216   3.33 0.000180  0.385  0.0232       3.33  3488
# 3 Yes       1013.   -5.26   0.203   3.25 0.000211  0.385  0.0203       3.25  5058
# 4 Yes        962.   -5.55   0.216   3.33 0.000180  0.385  0.0232       3.33  5709
# 5 No        2391.    2.55   0.242  -2.29 0.00395   0.387  0.0254      -2.30  5976

#### NOW RUN PREDICTIONS, BAE.
#lol. We can start by using the confusion matrix, which is a table that describes the classification performance 
#for each model on the test data.

#IMPORTANT. LIKE SUPER FREAKING IMPORTANT
# true positives (Bottom-right quadrant): 
# these are cases in which we predicted the customer would default and they did.
# true negatives (Top-left quadrant): 
# We predicted no default, and the customer did not default.
# false positives (Top-right quadrant): 
# We predicted yes, but they didn’t actually default. (Also known as a “Type I error.”)
# false negatives (Bottom-left): 
# We predicted no, but they did default. (Also known as a “Type II error.”)

test.predicted.m1 <- predict(model1, newdata = test, type = "response")
test.predicted.m2 <- predict(model2, newdata = test, type = "response")
test.predicted.m3 <- predict(model3, newdata = test, type = "response")

list(
  model1 = table(test$default, test.predicted.m1 > 0.5) %>% prop.table() %>% round(3),
  model2 = table(test$default, test.predicted.m2 > 0.5) %>% prop.table() %>% round(3),
  model3 = table(test$default, test.predicted.m3 > 0.5) %>% prop.table() %>% round(3)
)

#struggling to understand.
#in model 1, below, 96% are true-negatives. We predicted they would not default and they did not.
#in model 1, .01 percent predicted default and they did.
#in model 1, .025 (less than 3%) We predicted no, but they did default. (Also known as a “Type II error.”)
#in model 1, (Top-right quadrant): .003 We predicted yes, but they didn’t actually default. (Also known as a “Type I error.”)
#so I guess that 96% is good but not really understanding why 1% defaulted. seems like that would be more.
#actually, correct that model not that good, see later in tutorial. 
#model 2 = very accurate for not default, but no prediction for default.

# FALSE  TRUE
# No  0.962 0.003
# Yes 0.025 0.010
# 
# $model2
# 
# FALSE
# No  0.965
# Yes 0.035
# 
# $model3
# 
# FALSE  TRUE
# No  0.963 0.003
# Yes 0.026 0.009

