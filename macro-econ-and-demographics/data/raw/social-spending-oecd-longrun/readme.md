# Public social spending as share of GDP - Data package

This data package contains the data that powers the chart ["Public social spending as share of GDP"](https://ourworldindata.org/grapher/social-spending-oecd-longrun?v=1&csvType=full&useColumnShortNames=false) on the Our World in Data website. It was downloaded on May 25, 2026.

### Active Filters

A filtered subset of the full data was downloaded. The following filters were applied:

## CSV Structure

The high level structure of the CSV file is that each row is an observation for an entity (usually a country or region) and a timepoint (usually a year).

The first two columns in the CSV file are "Entity" and "Code". "Entity" is the name of the entity (e.g. "United States"). "Code" is the OWID internal entity code that we use if the entity is a country or region. For most countries, this is the same as the [iso alpha-3](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3) code of the entity (e.g. "USA") - for non-standard countries like historical countries these are custom codes.

The third column is either "Year" or "Day". If the data is annual, this is "Year" and contains only the year as an integer. If the column is "Day", the column contains a date string in the form "YYYY-MM-DD".

The final column is the data column, which is the time series that powers the chart. If the CSV data is downloaded using the "full data" option, then the column corresponds to the time series below. If the CSV data is downloaded using the "only selected data visible in the chart" option then the data column is transformed depending on the chart type and thus the association with the time series might not be as straightforward.


## Metadata.json structure

The .metadata.json file contains metadata about the data package. The "charts" key contains information to recreate the chart, like the title, subtitle etc.. The "columns" key contains information about each of the columns in the csv, like the unit, timespan covered, citation for the data etc..

## About the data

Our World in Data is almost never the original producer of the data - almost all of the data we use has been compiled by others. If you want to re-use data, it is your responsibility to ensure that you adhere to the sources' license and to credit them correctly. Please note that a single time series may have more than one source - e.g. when we stich together data from different time periods by different producers or when we calculate per capita metrics using population data from a second source.

## Detailed information about the data


## Public social expenditure as a share of GDP – Historical data – Lindert, OECD
Public social expenditure divided bt gross domestic product, expressed as a percentage.
Last updated: March 7, 2025  
Next update: June 2026  
Date range: 1880–2024  
Unit: % of GDP  


### How to cite this data

#### In-line citation
If you have limited space (e.g. in data visualizations), you can use this abbreviated in-line citation:  
OECD (2025); OECD (1985); Lindert (2004) – with major processing by Our World in Data

#### Full citation
OECD (2025); OECD (1985); Lindert (2004) – with major processing by Our World in Data. “Public social expenditure as a share of GDP – Lindert, OECD – Historical data” [dataset]. OECD, “OECD Social Expenditure Database (SOCX)”; OECD, “Social Expenditure 1960-1990: Problems of Growth and Control”; Lindert, “Growing Public: Social Spending and Economic Growth since the Eighteenth Century” [original data].
Source: OECD (2025), OECD (1985), Lindert (2004) – with major processing by Our World In Data

### What you should know about this data
* This indicator combines three different datasets: Lindert (2004), OECD (1985), and the OECD Social Expenditure Database (SOCX). We combine the two OECD datasets by using the implicit growth rate from the older series, so we can extend the series back to 1960. We also use the data from Lindert (2004) to extend the series to 1880.

### Sources

#### OECD – OECD Social Expenditure Database (SOCX)
Retrieved on: 2025-12-11  
Retrieved from: https://www.oecd.org/en/data/datasets/social-expenditure-database-socx.html  

#### OECD – Social Expenditure 1960-1990: Problems of Growth and Control
Retrieved on: 2025-03-07  
Retrieved from: https://archive.org/details/socialexpenditur0000unse  

#### Lindert – Growing Public: Social Spending and Economic Growth since the Eighteenth Century
Retrieved on: 2025-03-07  
Retrieved from: https://www.cambridge.org/core/books/growing-public/EAF17EB3BDFB5A6568930DBEC2CD1218  

#### Notes on our processing step for this indicator
We extrapolated the data available from the OECD Social Expenditure Database (public, in-cash and in-kind spending, all programs) using the earliest available observation from this dataset and applying the growth rates implied by the OECD (1985) data to obtain a series starting in 1960. These steps are necessary because the data in common years is not exactly the same for the two datasets due to changes in definitions and measurement. Nevertheless, we assume that trends stay the same in both cases.

We don't transform the data from Lindert (2004), the values are the same as in the original source.


    