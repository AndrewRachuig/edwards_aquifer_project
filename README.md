# The Edwards Aquifer and Bexar county: Looking to the Future

## The **WHY** of this project
As climate change increasingly affects how our planet, societies and economies function, water scarcity will hit everyone hard. Texas is no exception. The Texas Water Development Board estimates that at current trends "approximately 26 percent (13.3 million) of all Texans in 2070 would have less than half of the municipal water supplies they require." Further, without addressing these concerns "the estimated statewide impacts of not meeting the identified water needs in Texas would result in an annual combined lost income of \$110 billion in 2020, increasing to \$153 billion by 2070." See chapter 6 in the [link](https://www.twdb.texas.gov/waterplanning/swp/2022/docs/SWP22-Water-For-Texas.pdf) for more info.

It's imperative that we discover the causes, drivers, and various factors that play into available water. By truly understanding these now, we can properly implement mitigation strategies and prevention efforts before the situation becomes any more drastic than it already is.

This project is very limited in scope in that it attempts to understand a small slice of the picture in Texas. I have analyzed the historical water levels of Bexar county's J17 Index Well. The J17 gives a good snapshot of the day-to-day situation of the Edwards Aquifer as it pertains to available water for San Antonio and Bexar county at large. See [link](https://www.edwardsaquifer.net/j17.html) for more info. 

Ultimately, the goal is to predict future water levels through Time Series Analysis, but I've also explored relationships between historical water levels and other historical data (population, temperature, precipitation, water consumption) in an attempt to discover ways to better understand what may contribute to the Edwards Aquifer rise and fall. Pulling those variables in to make a more complex multivariate time series model would yield better predictions but it's beyond the scope of the current project.

# Project Summary
## Project Objectives
- Gain an understanding of historic water usage for Bexar county and what factors may affect rises or falls in the Edwards Aquifer water level.
    - Explore relationships between historic water level, water usage, population, precipitation, and temperature data for Bexar county.
- Perform a Time Series Analysis of the water level elevation for the Edwards Aquifer J17 Index Well.
    - Construct a resulting univariate time series model that accurately predicts water levels for J17 Index Well.
- Make relevant suggestions for future water supply conservation strategies.


## Audience
- Potential employers
- Fellow Data Science students as well as instructors at Codeup

---

## Project Deliverables
- A Final Report Notebook wherein I show how I arrived at the MVP model. Throughout the notebook, document conclusions, takeaways, and next steps.
    - Include [**Conclusions, recommendations, next steps**](#Conclusions,-Recommendations,-Next-Steps)
- A GitHub repo with final report
- A README.md containing a data dictionary, project and business goals, initial hypothesis and an executive summary for my github repo
- Project summary and writeup for my resume or other professional portfolio pieces
---

## Data Dictionary

|Target|Datatype|Definition|
|:-------|:--------|:----------|
| water_level_elevation| 31919 non-null: float64 | Edwards Aquifer J17 Index Well water elevation levels Nov 1932 - June 2022 |

source: https://www.edwardsaquifer.org/science-maps/aquifer-data/historical-data/

The following are additional features I pulled in from various sources to better understand the data.

|Feature|Datatype|Definition|
|:-------|:--------|:----------|
| avg_monthly_temp      | 1529 non-null: float64   |     The average monthly temperature for Bexar county Jan 1895 - May 2022 |
| total_monthly_precip  | 1529 non-null: float64   |     The total monthly precipitation for Bexar county Jan 1895 - May 2022  |

source: https://usafacts.org/issues/climate/state/texas/county/bexar-county

|Feature|Datatype|Definition|
|:-------|:--------|:----------|
| population           | 38 non-null: int64      |    Yearly population for Bexar county; prior to 2000 the population is given as an average for the decade  |

source: https://worldpopulationreview.com/us-counties/tx/bexar-county-population

|Feature|Datatype|Definition|
|:-------|:--------|:----------|
| total_consumption    | 36 non-null: int64      |    The total annual water use estimates for Bexar county from 1984 - 2019 |

source: https://www.twdb.texas.gov/waterplanning/waterusesurvey/estimates/index.asp

---
### Questions/thoughts I have of the Data
- I suspect there are predictable yearly variation patterns for the target variable, as well as larger seasons that I'm not yet aware of.
- Intuitively, I think population will affect water level as increased population will increase water usage and therefore decrease the water level.
- I also think precipitation amounts and water usage amounts will affect the water level. 
- It seems logical that precipitation, water usage, population, and water level would all be correlated to one another; unknown if there are causal relationships.
---
## Data Science Pipeline
#### Plan
- **Acquire** data from various sources:
    - Edwards Aquifer J17 Index Well Historical Data from: https://www.edwardsaquifer.org/science-maps/aquifer-data/historical-data/
    - Bexar County population data from US census data: https://worldpopulationreview.com/us-counties/tx/bexar-county-population
    - Historical Water Use estimates from Texas Water Development Board: https://www.twdb.texas.gov/waterplanning/waterusesurvey/estimates/index.asp
    - Bexar County historical climate data (temp and precipitation) from: https://usafacts.org/issues/climate/state/texas/county/bexar-county
    - Initial inquiry into the data to see the initial shape and layout.
- Clean and **prepare** data for the explore phase. Create wrangle.py to store functions I create to automate the full process of acquisition, cleaning and preparation. Separate train, validate, test subsets.
- Begin **exploration** of the data and ask questions leading to clarity of what is happening in the data.
    - Determine if any relationships exist that might explain historical water level elevation changes.
        - Find interactions between independent variables and the target variable using visualization and statistical testing.
    - Discover any trends and or seasonality in the target variable data.
- Create baseline **models** based on averages and then create more detailed predictive models. Compare performance and select best model.
    - Evaluate models on train and validate datasets. Do further hyperparameter tuning to find the best performing models.
    - Choose the model with that performs the best. Do any final tweaking of the model. Automate modeling functions and put them into a model.py file.
- Evaluate final model on the test dataset.
- Construct Final Report notebook wherein I show how I arrived at the final regression model by using my created modules. Throughout the notebook, document conclusions, takeaways, and next steps.
- Create README.md with data dictionary, project and business goals, initial hypothesis and an executive summary
---
#### Plan &rarr; Acquire
- Create wrangle.py to store all functions needed to acquire dataset (and later prepare/clean dataset).
- Retrieve the data and put it into a usable Pandas dataframe.
- Do cursory data exploration/summarization to get a general feel for the data contained in the dataset.
- Use the wrangle.py file to import and do initial exploration/summarization of the data in the Final Report notebook
---
#### Plan &rarr; Acquire &rarr; Prepare
- Explore the data further to see where/how the data is dirty and needs to be cleaned. This is not EDA. This is exploring individual variables so as to prepare the data to undergo EDA in the next step
- Use wrangle.py to store all functions needed to clean and prepare the dataset
    - A function which cleans the data:
        - Convert datatypes where necessary: objects to numerical; numerical to objects
        - Deal with missing values and nulls
        - Drop superfluous, erroneous or redundant data
        - Handle redundant categorical variables that can be simplified
        - Change names to snake case where needed
        - Drop duplicates
    - A function which splits the dataframe into 3 subsets: Train, and Test to be used for Exploration of the data in the next step
- Use the wrangle.py file to import and do initial cleaning/preparation of the data in the Final Report notebook
---
#### Plan &rarr; Acquire &rarr; Prepare &rarr; Explore
- Do Exploratory Data Analysis using bivariate and multivariate stats and visualizations to find interactions in the data
- Explore my key questions and discover answers to my hypotheses by running statistical analysis on data
- Explore the time series of the target variable to find patterns (trends, seasons, etc.) through decomposition of the data.
- Document all takeaways and answers to questions/hypotheses
- Create an explore.py file which will store functions made to aid in the data exploration
- Use explore.py and stats testing in the final report notebook to show how I arrived at my conclusions of the data
---
#### Plan &rarr; Acquire &rarr; Prepare &rarr; Explore &rarr; Model
- Do any final pre-modeling data prep (drop/combine columns) as determined most beneficial from the end of the Explore phase
- Find and establish baseline RSME based on various methods. Using the best of these "simple predictors" will give me an RSME level to beat with my models
- Create more complex predictive models to forecast future values of the target
    - Use Facebook Prophet as one of the more complex models
- For all models made, compare RSME results from train to validate
    - Look for hyperparameters that will give better results.
- Compare results from models to determine which is best. Chose the final model to go forward with
- Put all necessary functions for modeling functions into a model.py file
- Use model.py in the final report notebook to show how I reached the final model
- Having determined the best possible model, test it on out-of-sample data (the scaled test subset created in prepare) to determine predictive ability
- Summarize and visualize results. Document results and takeaways
---
#### Plan &rarr; Acquire &rarr; Prepare &rarr; Explore &rarr; Model &rarr; Deliver
- Go back over all code and make sure there are adequate comments on code
- Make sure the explantory text in the Final Report notebook properly conveys the information I am presenting
- Finish readme.md and other GitHub work to make the project presentable
---
# Executive Summary
- My best performing model, Facebook Prophet beat the baseline when predicting on out-of-sample data.
- Water level elevation as a target is highly variable, borderline random. This makes accurate predictions nearly impossible for univariate time series models.
- I found during Exploration that several of the datapoints I collected correlate to the target variable. 
    - Variables that I have shown to be statistically correlated with the target variable are **total water consumption and precipitation amounts**. Using these in a multivariate model could increase chances for better predictions, but In order to use these in a such a model it would require additional work not done in the current project.

**Recommendations**
- Despite beating baseline, I don't feel confident in this model. I would **not suggest** it be used going forward as it seems likely that it was random chance that the model beat baseline in this instance.
- Explore other reasons why the change in month-to-month water levels has grown increasingly variable over time post 1957/58.
- Further refine the work done here by creating a multivariate time series model.

## Reproduce this project
- To run through this project yourself you will need to copy all material listed in this GitHub repo. If desired, you could download the original data from the sources provided above in this readme.
- Otherwise, all other requisite files for running the final project notebook (all data files, script files, Jupyter notebook files) are contained in this repository.
