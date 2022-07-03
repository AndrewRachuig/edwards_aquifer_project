# Water and Bexar County: Looking to the Future

## The **WHY** of this project
As climate change increasingly affects how our planet, societies and economies work, water scarcity will hit everyone hard. Texas is no exception. The Texas Water Development Board estimates that at current trends "approximately 26 percent (13.3 million) of all Texans in 2070 would have less than half of the municipal water supplies they require." Further, without addressing these concerns "the estimated statewide impacts of not meeting the identified water needs in Texas would result in an annual combined lost income of \$110 billion in 2020, increasing to \$153 billion by 2070." See chapter 6 in the [link](https://www.twdb.texas.gov/waterplanning/swp/2022/docs/SWP22-Water-For-Texas.pdf) for more info.

It's imperative that we discover the causes, drivers, and various factors that play into available water. By truly understanding these now, we can properly implement mitigation strategies and prevention efforts before the situation becomes any more drastic than it already is.

This project is very limited in scope in that it attempts to understand a small slice of the picture in Texas. I have analyzed the historical water levels of Bexar county's J17 Index Well. The J17 gives a good snapshot of the day-to-day situation of the Edwards Aquifer as it pertains to available water for San Antonio and Bexar county at large. See [link](https://www.edwardsaquifer.net/j17.html) for more info.

# Overall Project Plan
## Goals
- Gain an understanding of historic water usage for Bexar county and what factors may affect rises or falls in the Edwards Aquifer water level.
    - Explore relationships between historic water level, water usage, population, precipitation, and temperature data for Bexar county.
- Perform a Time Series Analysis of the water level elevation for the Edwards Aquifer J17 Index Well.
    - Construct a resulting univariate time series model that accurately predicts water levels for J17 Index Well.
- Make relevant suggestions for future water supply conservation strategies.
---
## Data Science Pipeline

- [**Acquire**](#Data-Acquisition) data from various sources:
    - Edwards Aquifer J17 Index Well Historical Data from: https://www.edwardsaquifer.org/science-maps/aquifer-data/historical-data/
    - Bexar County population data from US census data: https://worldpopulationreview.com/us-counties/tx/bexar-county-population
    - Historical Water Use estimates from Texas Water Development Board:https://www.twdb.texas.gov/waterplanning/waterusesurvey/estimates/index.asp
    - Bexar County historical climate data (temp and precipitation) from: https://usafacts.org/issues/climate/state/texas/county/bexar-county
    - Initial inquiry into the data to see the initial shape and layout.
- Clean and [**prepare**](#Data-Preparation) data for the explore phase. Create wrangle.py to store functions I create to automate the full process of acquisition, cleaning and preparation. Separate train, validate, test subsets.
- Begin [**exploration**](#Data-Exploration) of the data and ask questions leading to clarity of what is happening in the data.
    - Determine if any relationships exist that might explain historical water level elevation changes.
        - Find interactions between independent variables and the target variable using visualization and statistical testing.
    - Discover any trends and or seasonality in the target variable data.
- Create baseline [**models**](#Modeling) based on averages and then create more detailed predictive models. Compare performance and select best model.
    - Evaluate models on train and validate datasets. Do further hyperparamter tuning to find the best performing models.
    - Choose the model with that performs the best. Do any final tweaking of the model. Automate modeling functions and put them into a model.py file.
- Evaluate final model on the test dataset.
---
## Deliverables
- Construct a Final Report Notebook wherein I show how I arrived at the MVP model. Throughout the notebook, document conclusions, takeaways, and next steps.
    - Include [**Conclusions, recomendations, next steps**](#Conclusions,-Recommendations,-Next-Steps)
- Deliver my github repo with final report to be analyzed by colleagues  
- Create README.md with data dictionary, project and business goals, initial hypothesis and an executive summary
- Project summary and writeup for my resume or other professional portfolio pieces


# Executive Summary
- My best performing model, Facebook Prophet beat the baseline when predicting on out-of-sample data.
- Water level elevation as a target is highly variable, borderline random. This makes accurate predictions nearly impossible for univariate time series models.
- I found during Exploration that several of the datapoints I collected correlate to the target variable. 
    - In order to use these in a predictive model it would require multivariate time series analysis which is is beyond the current scope of this project.
    - Variables that I have shown to be statistically correlated with the target variable are **total water consumption and precipitation amounts**. Using these in a multivariate model could increase chances for better predictions.

**Recommendations**
- Despite beating baseline, I don't feel confident in this model. I would **not suggest** it be used going forward as it seems likely that it was random chance that the model beat baseline in this instance.
- Further exploration of the data, finding additional correlating variables and discovering their trends and seasonality would be valuable in understanding any patterns in the target variable.
- Explore other reasons why the change in month-to-month water levels has grown increasingly variable over time post 1957/58.

**Future work**
- I would need further time to continue working towards creating a multivariate time series model. The additional work would  be significant and would require more study to adequately implement. However, the payoff could be considerable as it might prove to more accurately predict the target variable.
