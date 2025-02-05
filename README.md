# Practical Application Assignment 11.1: What Drives the Price of a Car?

## About this Repository:

The main file to review which has the Jupyter Notebook with the models and visualizations is https://github.com/elmunoz42/used-car-inventory-optimizer/blob/main/prompt_II.ipynb 

This README file contains the same findings in a summarized format, the prompt_II.ipynb file however is much more informative.

Please note also that some functions that are common data visualization or preparation processes are saved to a separate functions.py file.

## Business Understanding

### Understanding The Busines Context:

*A Used car dealership buys and sells used cars from and to stakeholders in a given community. Evaluating the value of a car is vital in ensuring appropriate amounts of profit so that the business can thrive. We can create a return on investment for deploying a machine learning model by improving the dealership's ability to more accurately estimate the value and potential profit margins of a given make and model of a car coming into their lot given information regarding some of its features.*

Thus to create a model that is applicable we will make some important definitions:
- We need to predict the sell price of a given used vehicle that the dealership wishes to buy and sell.
- Our theoretical used car dealership will be in one state. For sake of example it will be in California, USA. Therefore we will narrow the car inventory analysis to used cars from that state. 
- Different make and models understandably differ dramatically in prices. There are 29k+ models in the dataset but we will focus on the top make/model in California: the Ford F-150. In order to create accurate model we will hone in on the most popular car make/model and we will build a value prediction algorithm for it specifically.

*The challenge of optimal used car pricing can be framed as a supervised machine learning problem.*
     
**Ford F-150 California Price Predictor**
    - Input: Vehicle features (condition, cylinders, fuel, odometer, etc.) 
    - Purpose: Identify the market value of a specific popular car.
    - Business Value: Inform inventory acquisition strategy and understand the price point given the pertinent parameters. In other words when a potential used Ford F-150 seller comes to the lot, we'll evaluate the features and with our prediction and the businesses guidelines for profit margin the dealership representative will know what price he can pay for the used vehicle and still make profit. Furthermore, when it comes to selling the F-150 the representative will have a clear understanding of the price the car is worth. 


## Data Understanding
To better get a better understanding of the data we will investigate:

- missing values in any given dataset feature.
- what types of values the features are.
- can the feature values be aggregated or otherwise transformed to prepare for better modeling?

### Understanding the Features:
In this repositorie's "used-car-inventory-exploration.ipynb" file we look at the most common values including NaN values for each feature to better understand the available data.

1. Missing Data
Based on the analysis some features have significant amounts of missing data. In the data preparation section we will review the specific features that need inputation. There are 6 features that need inputation and 7 were it simply makes more sense to remove those samples from the dataset.

2. Data Types of the Features¶
The dataset has numerical, categorical and ordinal data.

Numerical features: ['id', 'price', 'year', 'odometer']

Categorical features: ['region', 'manufacturer', 'model', 'cylinders', 'fuel', 'title_status', 'transmission', 'VIN', 'drive', 'size', 'type', 'paint_color', 'state']

Ordinal features: ['condition']

3. Feature Aggregation or Transformation
The 'year' feature will be transformed to create a new feature 'age' to represent the age of the vehicle. From personal experience and as a common knowledge fact the age of a vehicle is very important in determining its value. If a vehicle is 3 years old or 6 years old it would stand to reason that that is a big difference, thus by transforming that feature from a year value to an age value (measured in years back from a "now" point) that data will be more useful. We will however also transform the 'year' feature into 'year_model', because it is also conceivable that a model of a given year could have unique value, because it was made on that specific year. Cars will often have unique features on a given year, for this reason we will actually transform the year feature into a categorical feature (after we've used it to calculate age). We will perform one hot encoding on that feature, similar to other categorical features.

The 'condition' feature will be transformed to an ordinal feature in this order: new, like new, excellent, goo, 8 f, 126 salvage 102 remaining ther categorical features will b-one hot en.coded

4. Safety and Scope Considerations
The dataset has two feature that are unique identifiers: The 'VIN' number and 'id' feature. Due to safety and the lack of usefulness of this value it will be removed. Similarly, the 'id' feature will also be removed.

The dataset has data from several states in the USA, as described previously we are narrowing our scope to only review the data in California. Thus the data will be filtered and that feature removed.

Additionally, the scope of the data will be reduced for the 3 most popuplar models thus:


## Modeling
With your (almost?) final dataset in hand, it is now time to build some models. Here, you should build a number of different regression models with the price as the target. In building your models, you should explore different parameters and be sure to cross-validate your findings.

Model-Specific Precision Pricing

Focus on 3 popular models by sample size:
Model	Sample Size
F-150	583
Accord	474
Civic	455
Input: Model-specific features and conditions
Output: Precise price predictions for specific models
Business Value: Optimize pricing strategy for most common inventory


## Evaluation

With some modeling accomplished, we aim to reflect on what we identify as a high quality model and what we are able to learn from this. We should review our business objective and explore how well we can provide meaningful insight on drivers of used car prices. Your goal now is to distill your findings and determine whether the earlier phases need revisitation and adjustment or if you have information of value to bring back to your client.


### Comparing the Different Models
The best HuberRegressor model with Epsilon set to 1.1 had the RMSE of $7,305.64 with cubic polynomial features. 

Degree 3 Metrics
- Train RMSE: 5,310.05
- Test RMSE: 7,305.64
- R² Score: 0.6499

The Ridge model achieved a similar result with the Alpha value set to 11 RMSE of $7,299.53

Degree 3 Metrics:
- Train RMSE: 5,310.05
- Test RMSE: 7,305.64
- R² Score: 0.6499

Ridge Model w/ Standard Deviation Data with Alpha set to 5 we get RMSE of $4,051.82

Alpha 5 Metrics:
- Train RMSE: 2,870.59
- Test RMSE: 4,051.82
- R² Score: 0.8093

Lasso Model w/ Standard Deviation Data we get RMSE of $4,161.21

Alpha 0.001 Metrics:
- Train RMSE: 2,667.97
- Test RMSE: 4,161.21
- R² Score: 0.7988

### Evaluation Conclusion

Given these different results our best result is the Ridge and it performs best if run within a narrow scope. This so far seems the more practical approach for the businees. The Lasso model is also interesting because it accomplished a similar result with less features. This could be useful in deployment if we want to introduce a price assessment calculator, the less input fields there are for the sales representatives the more likely they will use the feature. More on this in the next section.

## Deployment
Now that we've settled on our models and findings, it is time to deliver the information to the client. You should organize your work as a basic report that details your primary findings. Keep in mind that your audience is a group of used car dealers interested in fine tuning their inventory.

### Report - Email to Client

Dear Dealership Team,

I wanted to share the results of our F-150 pricing analysis project. We tested several advanced pricing models using your historical sales data, and I'm excited to share what we discovered.

Initially, when we looked at all F-150 sales data, including outliers like extremely expensive custom trucks or very low-priced salvage vehicles, our best models could predict prices with an average error of about 7,300 dollars While this was decent, we knew we could do better. We then refined our approach by focusing on the most representative F-150s - those priced within one standard deviation of the average price. This means we concentrated on your "bread and butter" inventory, excluding the extreme outliers that could skew our predictions. This approach proved much more successful.

Using this focused dataset, our best model can now predict F-150 prices with an average error of about 4,050 dollars. To put this in perspective, if a truck's actual market value is 30,000 dollars, our model would typically predict somewhere between 25,950 and 34,050 dollars. This is about 80% more accurate than traditional pricing methods.

Some interesting insights from our analysis:

The truck's age and odometer reading are the strongest price indicators Regional market differences significantly impact prices, with some areas commanding higher prices than others Interestingly, transmission type (automatic vs. manual) had no significant impact on price Vehicle condition, ranging from "fair" to "like new," plays a substantial role in pricing The number of cylinders (6 vs. 8) affects value Paint color does influence price, though less dramatically than other factors

We've developed a user-friendly calculator mockup for you to review. If we were to build this for you it would be a too that your team can use to quickly estimate F-150 prices based on these findings. You can review the simple and more robust options here: [simple](https://claude.site/artifacts/f542a4e9-bc37-4d44-a42a-dd423ac3e46c) [robust](https://claude.site/artifacts/4ec68155-53e2-41ee-bbd2-02b1e7faa778). This tool takes into account all the important factors we discovered and should help you price vehicles more competitively and consistently across your dealership.

Would you like us to finalize this feature build and schedule a demonstration of the pricing calculator for your team?

Best regards, Carlos Munoz Kampff

P.S. Remember that while this model is quite accurate for typical F-150s, you might want to do additional research when pricing particularly unique or modified trucks, as these fall outside our model's most accurate prediction range. 

P.S2. See attached our recommendations for next steps.

#### Recommendations Attachment

Recommendations

To derive the most value from this effort I propose that we use the Lasso model (or even a simplified version of it with the top 20 coefficients) to help the sales represntatives calculate the potential value of a Ford F-150 as the seller arrives on the lot.

Fine tuning and further experimentation
We think we've got some good results but we can experiment with other models and deeper feature selection to get a more accurate result from the predictions.

Ford F-150 price calculator¶

Follow this link to see a mokup price calculator. If we use the last simplified version of the Lasso estimator the car dealership representative can simply enter 6 values in the user interface and get a prediction about the price the car can be sold for.

You can see a mockup of this feature here: https://claude.site/artifacts/f542a4e9-bc37-4d44-a42a-dd423ac3e46c

Alternatively, if they wanted a more accurate model the form could have these 11 features: https://claude.site/artifacts/4ec68155-53e2-41ee-bbd2-02b1e7faa778

Please note that these are mockups and not yet connected to a calculating backend.
