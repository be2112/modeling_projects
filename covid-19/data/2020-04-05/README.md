# Data Description

## Overview

Covid-19 case, death, and population data by country.  The data is updated daily.

## Data Dictionary

| Variable                | Definition                                                   | Key                            |
| ----------------------- | ------------------------------------------------------------ | ------------------------------ |
| DateRep                 | Date that case data was reported                             |                                |
| day                     | Day of month                                                 | 0 = No, 1 = Yes                |
| month                   | Month                                                        | 1 = Jan, 2 = Feb, 3 = March... |
| year                    | Year                                                         |                                |
| cases                   | # of reported cases                                          |                                |
| deaths                  | \# of reported deaths                                        |                                |
| countriesAndTerritories | Full country/territory name                                  |                                |
| geoId                   | Country/Territory ID                                         |                                |
| countryterritoryCode    | International Standards Organization (ISO) 3-digit alphabetic code |                                |
| popData2018             | World Bank population data as of 2018                        |                                |

## Source

This data set was pulled from the European Centre for Disease Prevention and Control (https://opendata.ecdc.europa.eu/covid19/casedistribution/csv).

Population data is form the World Bank.

