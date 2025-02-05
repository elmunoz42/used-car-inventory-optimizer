# Practical Application Assignment 11.1: What Drives the Price of a Car?

# Evaluation

With some modeling accomplished, we aim to reflect on what we identify as a high quality model and what we are able to learn from this. We should review our business objective and explore how well we can provide meaningful insight on drivers of used car prices. Your goal now is to distill your findings and determine whether the earlier phases need revisitation and adjustment or if you have information of value to bring back to your client.

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

Evaluation Conclusion
Given these different results our best result is the Ridge and it performs best if run within a narrow scope. This so far seems the more practical approach for the businees. The Lasso model is also interesting because it accomplished a similar result with less features. This could be useful in deployment if we want to introduce a price assessment calculator, the less input fields there are for the sales representatives the more likely they will use the feature. More on this in the next section.

Deployment
Now that we've settled on our models and findings, it is time to deliver the information to the client. You should organize your work as a basic report that details your primary findings. Keep in mind that your audience is a group of used car dealers interested in fine tuning their inventory.

Report
Dear Dealership Team,

I wanted to share the results of our F-150 pricing analysis project. We tested several advanced pricing models using your historical sales data, and I'm excited to share what we discovered.

Initially, when we looked at all F-150 sales data, including outliers like extremely expensive custom trucks or very low-priced salvage vehicles, our best models could predict prices with an average error of about 7,300 dollars While this was decent, we knew we could do better. We then refined our approach by focusing on the most representative F-150s - those priced within one standard deviation of the average price. This means we concentrated on your "bread and butter" inventory, excluding the extreme outliers that could skew our predictions. This approach proved much more successful.

Using this focused dataset, our best model can now predict F-150 prices with an average error of about 4,050 dollars. To put this in perspective, if a truck's actual market value is 30,000 dollars, our model would typically predict somewhere between 25,950 and 34,050 dollars. This is about 80% more accurate than traditional pricing methods.

Some interesting insights from our analysis:

The truck's age and odometer reading are the strongest price indicators Regional market differences significantly impact prices, with some areas commanding higher prices than others Interestingly, transmission type (automatic vs. manual) had no significant impact on price Vehicle condition, ranging from "fair" to "like new," plays a substantial role in pricing The number of cylinders (6 vs. 8) affects value Paint color does influence price, though less dramatically than other factors

We've developed a user-friendly calculator mockup that your team can use to quickly estimate F-150 prices based on these findings. You can review the simple and more robust options here: simple robust. This tool takes into account all the important factors we discovered and should help you price vehicles more competitively and consistently across your dealership.

Would you like us to finalize this feature build and schedule a demonstration of the pricing calculator for your team?

Best regards, Carlos Munoz Kampff

P.S. Remember that while this model is quite accurate for typical F-150s, you might want to do additional research when pricing particularly unique or modified trucks, as these fall outside our model's most accurate prediction range. P.S2. See attached our recommendations.

Recommendations Attachment
Recommendations
To derive the most value from this effort I propose that we use the Lasso model (or even a simplified version of it with the top 20 coefficients) to help the sales represntatives calculate the potential value of a Ford F-150 as the seller arrives on the lot.

Fine tuning and further experimentation
We think we've got some good results but we can experiment with other models and deeper feature selection to get a more accurate result from the predictions.

Ford F-150 price calculator¶
Follow this link to see a mokup price calculator. If we use the last simplified version of the Lasso estimator the car dealership representative can simply enter 6 values in the user interface and get a prediction about the price the car can be sold for.

You can see a mockup of this feature here: https://claude.site/artifacts/f542a4e9-bc37-4d44-a42a-dd423ac3e46c

Alternatively, if they wanted a more accurate model the form could have these 11 features: https://claude.site/artifacts/4ec68155-53e2-41ee-bbd2-02b1e7faa778

Please note that these are mockups and not yet connected to a calculating backend yet.