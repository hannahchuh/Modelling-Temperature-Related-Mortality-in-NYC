# Modelling Mortality and Temperature in NYC
Given EX1, EX2, and EX3, the relationship between has become a rising global public health concern. How will increasing temperatures due to climate change affect mortality in the future? This research attempts to investigate such a relationship. The research was conducted in the Summer/Fall of 2017, winning first in Earth and Environmental Sciences at the Western Nevada Science and Engineering Fair and an Intel ISEF Finalist. The code, full paper, research prsentation poster, and some sample data are hosted here. I have analyzed several decades of temperature, dew point, and mortality data from New York City, using least-squares linear regression to model future heat-related mortalities in the future. My findings demonstrate a significant and concerning increase in annual temperature-related mortalities.

<h1>Data Sources</h1>

- <b>Mortality Data: </b>Observational mortality data from the Health Effects Institute Research Report 94, 1987-2000

- <b>Climate Data: </b>Dew points and temperatures (hourly basis) from a National Oceanic and Atmospheric Administration, LaGuardia Airport
  - 987-2000 period for training the model
  - 1973-2015 for additional analysis

- <b>Climate Projections: </b>Seven GCMs from CMIP5 for temperature and dew point projections, 1988-2000 and 2020-2080
  - BNU-ESM
  - CSIRO-MK3-6-0
  - CNRM-CM5
  - GFDL-CM3
  - IPSL-CM5A-MR
  - NORESM1-M

<h1>Tools</h1>

- scipy

- scikit-learn

- matplotlib

- pickle

<h1>The offical abstract:</h1>
  
Heatwaves cause significant increases in the average daily mortality of a region and thus pose a serious and growing public health risk, particularly in the context of anthropogenic climate change (Kalkstein & Greene, 1997; Kovats & Hajat, 2008; Meehl & Tebaldi, 2004). Although net winter mortality currently exceeds that of summer, rising global average temperatures will cause increases in heat-related mortality that will not be offset by declines in cold-related mortality. This study uses temperature, dew point, and mortality data from 1987 to 2000 in New York City to develop a model projecting daily temperature-related mortality anomalies and predict how climate change may affect daily mortality in the 21st century. The resulting model was run on seven general circulation models from the years 2020-2080; an analysis of the developed model’s projections shows a significant overall increase in annual temperature-related mortality, suggesting a need to address the rising risk of extreme heat caused by climate change.

<h1>Poster</h1>

![Image of Poster](https://raw.githubusercontent.com/hannahchuh/Modelling-Temperature-Related-Mortality-in-NYC/master/Poster.png)
