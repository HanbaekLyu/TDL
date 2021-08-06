# JHU CSSE COVID-19 Dataset

## Table of contents

 * [Daily reports (csse_covid_19_daily_reports)](#daily-reports-csse_covid_19_daily_reports)
 * [USA daily state reports (csse_covid_19_daily_reports_us)](#usa-daily-state-reports-csse_covid_19_daily_reports_us)
 * [Time series summary (csse_covid_19_time_series)](#time-series-summary-csse_covid_19_time_series)
 * [Data modification records](#data-modification-records)
 * [UID Lookup Table Logic](#uid-lookup-table-logic)
---

## [Daily reports (csse_covid_19_daily_reports)](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports)

This folder contains daily case reports. All timestamps are in UTC (GMT+0).

### File naming convention
MM-DD-YYYY.csv in UTC.

### Field description
* <b>FIPS</b>: US only. Federal Information Processing Standards code that uniquely identifies counties within the USA.
* <b>Admin2</b>: County name. US only.
* <b>Province_State</b>: Province, state or dependency name.
* <b>Country_Region</b>: Country, region or sovereignty name. The names of locations included on the Website correspond with the official designations used by the U.S. Department of State.
* <b>Last Update</b>: MM/DD/YYYY HH:mm:ss  (24 hour format, in UTC).
* <b>Lat</b> and <b>Long_</b>: Dot locations on the dashboard. All points (except for Australia) shown on the map are based on geographic centroids, and are not representative of a specific address, building or any location at a spatial scale finer than a province/state. Australian dots are located at the centroid of the largest city in each state.
* <b>Confirmed</b>: Counts include confirmed and probable (where reported).
* <b>Deaths</b>: Counts include confirmed and probable (where reported).
* <b>Recovered</b>: Recovered cases are estimates based on local media reports, and state and local reporting when available, and therefore may be substantially lower than the true number. US state-level recovered cases are from [COVID Tracking Project](https://covidtracking.com/).
* <b>Active:</b> Active cases = total cases - total recovered - total deaths.
* <b>Incidence_Rate</b>: Incidence Rate = cases per 100,000 persons.
* <b>Case-Fatality Ratio (%)</b>: Case-Fatality Ratio (%) = Number recorded deaths / Number cases.

### Update frequency
* Since June 15, We are moving the update time forward to occur between 04:45 and 05:15 GMT to accommodate daily updates from India's Ministry of Health and Family Welfare.
* Files on and after April 23, once per day between 03:30 and 04:00 UTC. 
* Files from February 2 to April 22: once per day around 23:59 UTC.
* Files on and before February 1: the last updated files before 23:59 UTC. Sources: [archived_data](https://github.com/CSSEGISandData/COVID-19/tree/master/archived_data) and dashboard.

### Data sources
Refer to the [mainpage](https://github.com/CSSEGISandData/COVID-19).

### Why create this new folder?
1. Unifying all timestamps to UTC, including the file name and the "Last Update" field.
2. Pushing only one file every day.
3. All historic data is archived in [archived_data](https://github.com/CSSEGISandData/COVID-19/tree/master/archived_data).

---
## [USA daily state reports (csse_covid_19_daily_reports_us)](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports_us)

This table contains an aggregation of each USA State level data.

### File naming convention
MM-DD-YYYY.csv in UTC.

### Field description
* <b>Province_State</b> - The name of the State within the USA.
* <b>Country_Region</b> - The name of the Country (US).
* <b>Last_Update</b> - The most recent date the file was pushed.
* <b>Lat</b> - Latitude.
* <b>Long_</b> - Longitude.
* <b>Confirmed</b> - Aggregated case count for the state.
* <b>Deaths</b> - Aggregated death toll for the state.
* <b>Recovered</b> - Aggregated Recovered case count for the state.
* <b>Active</b> - Aggregated confirmed cases that have not been resolved (Active cases = total cases - total recovered - total deaths).
* <b>FIPS</b> - Federal Information Processing Standards code that uniquely identifies counties within the USA.
* <b>Incident_Rate</b> - cases per 100,000 persons.
* <b>People_Tested</b> - Total number of people who have been tested.
* <b>People_Hospitalized</b> - Total number of people hospitalized.
* <b>Mortality_Rate</b> - Number recorded deaths * 100/ Number confirmed cases.
* <b>UID</b> - Unique Identifier for each row entry. 
* <b>ISO3</b> - Officialy assigned country code identifiers.
* <b>Testing_Rate</b> - Total test results per 100,000 persons. The "total test results" are equal to "Total test results (Positive + Negative)" from [COVID Tracking Project](https://covidtracking.com/).
* <b>Hospitalization_Rate</b> - US Hospitalization Rate (%): = Total number hospitalized / Number cases. The "Total number hospitalized" is the "Hospitalized – Cumulative" count from [COVID Tracking Project](https://covidtracking.com/). The "hospitalization rate" and "Total number hospitalized" is only presented for those states which provide cumulative hospital data.

### Update frequency
* Once per day between 04:45 and 05:15 UTC.

### Data sources
Refer to the [mainpage](https://github.com/CSSEGISandData/COVID-19).

---
## [Time series summary (csse_covid_19_time_series)](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series)

See [here](https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/README.md).

---
## Data modification records
We are also monitoring the curve change. Any errors made by us will be corrected in the dataset. Any possible errors from the original data sources will be listed here as a reference.
* NHC 2/14: Hubei Province deducted 108 prior deaths from the death toll due to double counting.
* For Hubei Province: from Feb 13 (GMT +8), we report both clinically diagnosed and lab-confirmed cases. For lab-confirmed cases only (Before Feb 17), please refer to [who_covid_19_situation_reports](https://github.com/CSSEGISandData/COVID-19/tree/master/who_covid_19_situation_reports). 
* On Feb 27 Italy made a change in their testing protocols, to limit coronavirus testing to at-risk people showing symptoms of COVID-19. ([Source](https://apnews.com/6c7e40fbec09858a3b4dbd65fe0f14f5))
* About DP 3/1: All cases of COVID-19 in repatriated US citizens from the Diamond Princess are grouped together, and their location is currently designated at the ship’s port location off the coast of Japan. These individuals have been assigned to various quarantine locations (in military bases and hospitals) around the US. This grouping is consistent with the CDC.
* Hainan Province active cases update (4/13): We responded to the error from 3/24 to 4/1 we had incorrect data for Hainan Province.  We had -6 active cases (168 6 168 -6). We applied the correction (168 6 162 0) that was applied on 4/2 for this period (3/24 to 4/1).
* Florida in the daily report US (4/13): Source data error. Correction 123,019 -> 21,019.
* Okaloosa, Florida in the dail report (4/13): Source data error. Correction 102,103 -> 103.
* The death toll in Wuhan was revised from 2579 to 3869 (4/17). ([Source1](http://www.china.org.cn/china/Off_the_Wire/2020-04/17/content_75943843.htm), [Source2](http://www.nhc.gov.cn/yjb/s7860/202004/51706a79b1af4349b99264420f2cee54.shtml))
* About France confirmed cases (4/16): after communicating with solidarites-sante.gouv.fr, we decided to make these adjustments based on public available information. From April 4 to April 11, only "cas confirmés" are counted as confirmed cases in our dashboard. Starting from April 12, both "cas confirmés" and "cas possibles en ESMS" (probable cases from ESMS) are counted into confirmed cases in our dashboard. ([More details](https://github.com/CSSEGISandData/COVID-19/issues/2094))
* Benton and Franklin, WA on April 21 and 22. Data were adjusted/added to match the WA DOH report. See [errata](https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/Errata.csv) for details.
* April 22, cases within the Navajo Nation had been tracked as an independent data source which resulted in double counting of the cases and deaths within Arizona, New Mexico, and Nevada. The US time series files for confirmed from 4/1 and 4/8 and the US time series files for deaths from 3/31 to 4/17 were corrected to remove the double counting. Adjustments were also made for Navajo County, AZ; Cococino County, AZ; Apache County, AZ; San Juan County, NM; McKinley County, NM; Cibola County, NM; Socorrco County, NM; and San Juan County, UT. See errata file for specfic details.
* April 24, time_series_covid19_deaths_us.csv for New York City, NY adjusted to add back-distribute dated probable deaths. time_series_covid19_confirmed_us.csv for New York City, NY adjusted to remove probable deaths as cases. This change is in line with CDC reporting guidelines.
* April 26, recovered data for Australian territories from 4/20 to 4/26 had gone stale. Historical values from new source used to fill in the six day gap.
* April 28, for consistency, we no longer report the hospitalization data as the max of "current - hospitalized" and "cumulative - hospitalized", and instead only report 'cumulative - hospitalized' from [Covid Tracking Project](https://covidtracking.com/). For states that do not provide cumulative hospital counts no hospital data will be shown.
* April 28, Lithuania: The number of confirmed infection cases. Until 28 April, information has been provided on positive laboratory test results rather than on positive cases (people). ([Source](https://lietuva.lt/wp-content/uploads/2020/04/UPDATE-April-28.pdf)).
* April 30, all death values for the United Kingdom adjusted due to the release of deaths in care homes.
* May 1, all data for Kosovo and Serbia from 4/19-4/30 adjusted due to stale data source.
* May 2, clarification of the handling of data in France ([GitHub Issue](https://github.com/CSSEGISandData/COVID-19/issues/2459)).
* May 15, clairification of the handling of data for Spain ([GitHub Issue](https://github.com/CSSEGISandData/COVID-19/issues/2522)).
* May 20, the drop of cumulative confirmed cases in the UK. "This is due to historical data revisions across all pillars." ([Source](https://www.gov.uk/guidance/coronavirus-covid-19-information-for-the-public), [DHSCgovuk Twitter](https://twitter.com/DHSCgovuk/status/1263159710892638208)).
* May 27, removal of recovered data from Netherlands due to lack of data reporting by national health ministry.
* June 2, France: Reduction in confirmed cases due to a change in calculation method. Since June 2, patients who test positive are only counted once. (Baisse des cas confirmés due à un changement de méthode de calcul. Depuis le 2 juin, les patients testés positifs ne sont plus comptés qu’une seule fois.) ([Source](https://dashboard.covid19.data.gouv.fr/vue-d-ensemble?location=FRA))
* June 5, On June 2nd, Chile’s Ministerio de Salud began reporting national “total active cases” where in the past they had reported national “total recoveries”.  To accommodate this change and to stay consistent with the ministry’s reporting of active cases, from June 2nd forward we are computing recoveries based on the formula “Active Cases = Total Case – Deaths – Recoveries”.  Based on this, the data for Chile will reflects a jump in recoveries on June 2nd. ([Source](https://www.minsal.cl/nuevo-coronavirus-2019-ncov/casos-confirmados-en-chile-covid-19/))
* June 5, In an internal audit of the data for Sweden, it has become clear to our team that our reported total of recoveries conflates regional reporting of the number of patients being released from hospitals with country wide recovery data.  As this regional reporting is not universally available and represents only a subset of recoveries, our prior reporting did not accurately represent nationwide recoveries.  To ensure the accuracy of our data, we have chosen to nullify the number of recovered cases in Sweden until the data is released by the national health ministry. We will also be removing recovery data from our historical time series due to this assessment.
* June 5, As noted in the disclaimer for the dashboard, the geographic designations in this data have been designed to be consistent with public guidance from the US State Department.  This does not imply the expression of any opinion whatsoever on the part of JHU concerning the legal status of any country, area or territory or of its authorities.  In implementing subnational data for the Russian Federation and the Ukraine, data for the Crimean Peninsula has been apportioned in line with this guidance.  This adjustment explains a difference in national totals for both the Russian Federation and Ukraine relative to alternate reporting.
* June 10, Our previous reporting for Pakistan had a single day delay. A recent update corrected this issue but resulted in data for June 7th being lost. We have corrected this issue by adding June 7th manually and pulling all of the Pakistan data back by a single day.
* June 11, Michigan, US. Michigan started to report probable cases and probable deaths on June 5. ([Source](https://www.michigan.gov/coronavirus/0,9753,7-406-98158-531156--,00.html)) We combined the probable cases into the confirmed cases, and the probable deaths into the deaths. As a consequence, a spike with 5.5k+ cases is shown in our daily cases bar chart.
* June 13, Through data provided by the Michigan Department of Health and Human Service’s (MDHHS) Communicable Disease Division, we were able to appropriately distribute the probable cases MDHHS began reporting on June 5th.
* June 12, Louis City, MO, data for confirmed cases and deaths from March 16 to June 11 were updated to match up with the updated official report at the [City of St. Louis dashboard](https://www.stlouis-mo.gov/covid-19/data/index.cfm). Date of the first case was updated to March 16, and date of the first deaths was updated to March 23.
* June 12, Louis County, MO, data for confirmed cases and deaths from March 9 to June 11 were updated to match up with the updated official report at [St. Louis County government site](https://stlcorona.com/resources/covid-19-statistics1/). Date of the first case was remained on March 8, and date of the first deaths was updated to March 20.
* June 12, Massachusetts, cases from April 15 to June 11 were updated to match official updateded statistics from the [Massachusetts government raw data - County.csv](https://www.mass.gov/info-details/covid-19-response-reporting). This change arose due to to release of historical probable cases by the state. The alteration distributes probable cases and updates some confirmed case counts that were revised by the state. Dukes and Nantucket are still reported together, though County.csv lists them separately.
* June 16th, delay in reporting from Oregon Health Authority resulted in time series for confirmed and deaths not updating for June 14th. Updated via data from this [report](https://www.oregon.gov/oha/ERD/Pages/Oregon-reports-101-new-confirmed-presumptive-COVID-19-cases-2-new-deaths.aspx). Recovered data was not available for this date.
* June 19th, cases data for Belarus on April 18th and 19th were adjusted. Initial error was due to a [delay] (https://news.tut.by/society/681391.html) in reporting by the Belarusian health authorities that wasn't properly distributed.
* Not a data modification, but we wish to draw attention to [issue #2722](https://github.com/CSSEGISandData/COVID-19/issues/2722) that explains the recent spike in cases in Chile.
* June 25, NJ began reporting probable deaths today and the record for the 25th reflects these 1854 deaths not previously reported.  Additional information can be found in the [transcript](https://nj.gov/governor/news/news/562020/approved/20200625a.shtml) of the state's June 25th coronavirus briefing.
* June 27, internal audit identified issue with calculation of probable cases in nursing homes for France. The French Health Ministry ended public reporting of this number on June 1st - we have since carried that number of probable cases forward.
* June 30, the count of New Yorkers who have died of COVID-19 increased by 692. ([NYC gov](https://www1.nyc.gov/site/doh/covid/covid-19-data.page)) We distributed these data back to the time series tables according to [nychealth GitHub](https://github.com/nychealth/coronavirus-data/blob/master/deaths/probable-confirmed-dod.csv).
* July 3rd, on July 2nd, the United Kingdom revised their case count due to double counting of cases in England that had been tested in multiple facilities. In doing so, they revised their historical time series data for England (available [here](https://coronavirus.data.gov.uk/)). This change resulted in the need to revise our time series for the United Kingdom. As our time series data represents collective cases in England, Scotland, Northern Ireland, and Wales and the change only affected England, we gathered historical from each respective country's national dashboard (available [here](https://public.tableau.com/profile/public.health.wales.health.protection#!/vizhome/RapidCOVID-19virology-Public/Headlinesummary), [here](https://www.arcgis.com/apps/opsdashboard/index.html#/658feae0ab1d432f9fdb53aa082e4130), and [here](https://app.powerbi.com/view?r=eyJrIjoiZGYxNjYzNmUtOTlmZS00ODAxLWE1YTEtMjA0NjZhMzlmN2JmIiwidCI6IjljOWEzMGRlLWQ4ZDctNGFhNC05NjAwLTRiZTc2MjVmZjZjNSIsImMiOjh9)) to completely rewrite the time series data for cases in the United Kingdom.
* July 9, Japan's data were updated according to the [Japan COVID-19 Coronavirus Tracker](https://covid19japan.com/). Confirmed cases were updated from Feb 5 to May 27, and deaths were updated from Feb 13 to May 27.
* July 14, United Kingdom death data has historical revisions. Death data was downloaded from [this link](https://coronavirus.data.gov.uk/downloads/csv/coronavirus-deaths_latest.csv) and the death totals for the UK from 3/25 to 6/22 in time_series_covid19_deaths_global.csv were updated to match the data in the official report.
* July 18, we are now providing the confirmed cases for Puerto Rico at the municipality (Admin1) level. The historic Admin1 data ranging from 5/6 to 7/17 are from [nytimes dataset](https://github.com/nytimes/covid-19-data). Confirmed cases before 5/6 are categorized into Unassigned, Puerto Rico in `time_series_covid19_confirmed_US.csv`. Meanwhile, deaths are all grouped into Unassigned, Puerto Rico in `time_series_covid19_deaths_US.csv`. Daily cases are from [Puerto Rico Departamento de Salud](http://www.salud.gov.pr/Pages/coronavirus.aspx).
* July 20, the negative active cases in Uganda is due to different criteria. According to a tweet mentioned by Uganda Ministry of Health, the recovered cases include Ugandans, non Ugandans and refugees while confirmed cases capture only Ugandans. ([source](https://twitter.com/gbkatatumba/status/1285150623692926976))
* July 22nd, updates to Liechtenstein cases and recovered in line with historical data provided on this [government website](https://www.llv.li/inhalt/118863/amtsstellen/situationsbericht) and within this [pdf](https://www.llv.li/files/ag/aktuelle-fallzahlen.pdf)
* July 22, update Iceland confirmed cases and recovered cases (June 15 to July 20) according to the Directorate of Health and the Department of Civil Protection and Emergency Management of Iceland (https://www.covid.is/data). The positive with antibodies instances no longer figured into the number of total cases.
* July 26, overwrite Chile time series data to distribute probable and non-notified cases occuring prior to June 17th. Data from [this repository] (https://github.com/MinCiencia/Datos-COVID19) managed by the Ministry of Science was used for the correction. Specifically, data from [product 45] CasosConfirmadosPorComunaHistorico_std.csv and CasosNoNotificadosPorComunaHistorico_std.csv was accessed on July 26 and the most current version of the documents at that time were used for the correction. For CasosConfirmadosPorComunaHistorico_std.csv, this was July 22nd. Cases were added to the day at the end of their respective epidemiological week. 
* July 28, Data for Kosovo was revised based on reporting from the [Kosovo National Institute of Public Health](https://www.facebook.com/IKSHPK), the [Kosovo Corona Tracker](https://corona-ks.info/?lang=en), and coincident reporting from local news sources: [Koha Ditore](https://www.koha.net/) and [Telegrafi](https://telegrafi.com/).  Data was updated from 3/14 to 7/26.

## Retrospective reporting of (probable) cases and deaths
This section reports instances where large numbers of historical cases or deaths have been reported on a single day. These reports cause anomalous spikes in our time series curves. When available, we liaise with the appropriate health department and distribute the cases or deaths back over the time series. A large proportion of these spikes are due to the release of probable cases or deaths.
* April 12, The spike in France confirmed cases on April 12 is due to the new inclusion of "cas possibles en ESMS" (probable cases from ESMS), which are counted into confirmed cases in our dashboard. ([More details](https://github.com/CSSEGISandData/COVID-19/issues/2094))
* April 21, deaths in Finland rise from 98 to 141 as the Finnish National Institute for Health and Welfare included deaths in nursing homes in the Helsinki Metropolitan area for the first time. ([Source](https://www.foreigner.fi/articulo/coronavirus/finland-reports-44-increase-in-number-of-coronavirus-deaths/20200421174642005414.html))
* April 23, New York City began reporting probable deaths ([source](https://www.nbcnews.com/health/health-news/live-blog/2020-04-23-coronavirus-news-n1190201/ncrd1190406#blogHeader)). The large number of deaths were back distributed through March 12th (see errata.csv line 104).
* April 24, Colorado began reporting probable deaths. This resulting in a spike of 121 probable deaths on that day ([source](https://www.denverpost.com/2020/04/24/covid-coronavirus-colorado-new-cases-deaths-april-24/)).
* April 24, the Republic of Ireland begins including probable deaths (those with COVID-19 listed as cause of death but no molecular test). 189 probable deaths are added to death total. ([Source](https://www.irishnews.com/news/republicofirelandnews/2020/04/24/news/republic-s-covid-19-death-toll-passes-1-000-1915278/))
* April 29, United Kingdom updated death counts to reflect deaths outside of hospitals ([source](https://metro.co.uk/2020/04/29/uk-death-toll-rises-26097-care-homes-included-12628454/)). Corrections were made to match time series on official website.
* May 6, Belgium reports 339 new deaths, 229 of which had occured over recent weeks. ([Source](http://www.xinhuanet.com/english/2020-05/06/c_139035611.htm))
* June 5th, as reported in [Issue #2704](https://github.com/CSSEGISandData/COVID-19/issues/2704), the state of Michigan released probable cases and probable deaths. The probable cases and deaths have been distributed over March 12th and June 10th as advised by the Michigan Department of Health and Human Services.
* June 12, probable cases in Massachusetts redistributed over time series. File source is [here](https://www.mass.gov/info-details/covid-19-response-reporting).
* June 16, Spain releases revision to death count that resulted in a spike of 1179 deaths [source 1](https://www.mscbs.gob.es/profesionales/saludPublica/ccayes/alertasActual/nCov-China/documentos/Actualizacion_140_COVID-19.pdf) & [source 2](https://www.mscbs.gob.es/profesionales/saludPublica/ccayes/alertasActual/nCov-China/documentos/Actualizacion_141_COVID-19.pdf)
* June 16, India releases 2004 deaths, 1672 of which were backlogged deaths from Delhi and Maharashtra [source](https://www.hindustantimes.com/india-news/india-s-death-toll-soars-past-10k-backlog-deaths-raise-count-by-437-in-delhi-1-409-in-maharashtra/story-9GNbe7iMBKLsiHtByjRKCJ.html).
* June 17, as reported in [Issue #2722](https://github.com/CSSEGISandData/COVID-19/issues/2722), the country of Chile released 31k cases that had previously been absent from their national counts. This data has been corrected as noted above in the Data Modification records.
* June 22, The spike in cases (and deaths) in Mississippi on June 22 is due to a lack of reporting by the state for the four days prior due to a reported technical issue (https://msdh.ms.gov/msdhsite/_static/resources/8675.pdf).
* June 23, as reported in [Issue #2789](https://github.com/CSSEGISandData/COVID-19/issues/2789), the state of Delaware released some probable deaths and identified historical confirmed deaths. We are actively engaged with stakeholders to determine how to distribute these deaths over time.
* June 25, as reported in [Issue #2763](https://github.com/CSSEGISandData/COVID-19/issues/2763), the state of New Jersey released probable deaths. The probable deaths were redistributed to the "Unassigned, New Jersey" entry of time_series_covid19_deaths_US.csv on August 2.
* July 1, the count of New Yorkers who have died of COVID-19 increased by 692 on June 30. ([NYC gov](https://www1.nyc.gov/site/doh/covid/covid-19-data.page)) On July 1 we distributed these data back to the time series tables according to [nychealth GitHub](https://github.com/nychealth/coronavirus-data/blob/master/deaths/probable-confirmed-dod.csv).
* July 7, incorporation of probable cases and deaths that are being released by the Illinois Department of Health once per week, starting July 3rd. We anticipate weekly spikes in both of these numbers.
* July 12, the Philippines reports 162 deaths, 90 of which occurred in June and 51 occurred on other dates in July. ([Source](https://rappler.com/nation/coronavirus-cases-philippines-july-12-2020))
* July 17, the governments of Kazakhstan and Kyrgyzstan alter their definition of probable case to include those diagnosed with pneumonia that have not been tested for COVID-19. This change in definition results in spikes in deaths of Kyrgyzstan on July 18 and Kazakhstan on July 29. ([Source](https://www.hrw.org/news/2020/07/21/kyrgyzstan/kazakhstan-new-rules-tallying-covid-19-data))
* July 22, Peru added an additional 3688 deaths from analyzing historical death records. It is unclear if these are probable deaths or retroactively diagnosed. [Source](https://www.gob.pe/institucion/minsa/noticias/214828-minsa-casos-confirmados-por-coronavirus-covid-19-ascienden-a-366-550-en-el-peru-comunicado-n-180)
* July 27, The state of Texas' Department of State Health Services changed their reporting methodology for COVID-19 deaths, resulting in a roughly 13% increase in reported fatalities from the 26th to the 27th.  Details can be found in the press release from the state [here](https://www.dshs.texas.gov/news/releases/2020/20200727.aspx).
* July 29, Connecticut cases rise by 463 cases. 384 cases of the cases are from lab tests "performed during April-June (which) were newly reported to DPH in connection with a transition to electronic reporting by an out of state regional laboratory and for surveillance purposes have been added to the total case and test counts" ([source](https://portal.ct.gov/-/media/Coronavirus/CTDPHCOVID19summary7292020.pdf)). The 463 spike is consistent with the ct.gov data ([source](https://data.ct.gov/Health-and-Human-Services/COVID-19-Tests-Cases-Hospitalizations-and-Deaths-S/rf3k-f8fg/data)).

## Irregular Update Schedules
As the pandemic has progressed, several locations have altered their reporting schedules to no longer provide daily updates. As these locations are identified, we will list them in this section of the README. We anticipate that these irregular updates will cause cyclical spikes in the data and smoothing algorithms should be applied if the data is to be used for modeling.

United States
* Rhode Island: Not updating case, death, or recovered data on the weekends
* Conneticut: Not updating case, death, or recovered data on the weekends
* Illinois: Releasing probable cases once per week.
* District of Columbia: No weekend update for the first week of August.
* Louisiana: No weekend update for the first week of August.

International
* Sweden: Not updating case, death, or recovered data on the weekends
* Spain: Not updating case or death data on the weekends (and is not currently providing recoveries at any time)
* Nicaragua: Releasing case, death, and recovered data once per week.
* UK: daily death toll paused on July 18. ([GOV.UK](https://www.gov.uk/guidance/coronavirus-covid-19-information-for-the-public#number-of-cases) and [Reuters](https://www.reuters.com/article/us-health-coronavirus-britain-casualties-idUSKCN24J0GC))
* France: No longer releasing case, hospitalization, or death data on the weekends. Please see [Tableau dashboard](https://dashboard.covid19.data.gouv.fr/vue-d-ensemble?location=FRA). 
* Denmark: Not updating case, death, or recovered data on the weekends.

---
## [UID Lookup Table Logic](https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv)

1.	All countries without dependencies (entries with only Admin0).
  *	None cruise ship Admin0: UID = code3. (e.g., Afghanistan, UID = code3 = 4)
  *	Cruise ships in Admin0: Diamond Princess UID = 9999, MS Zaandam UID = 8888.
2.	All countries with only state-level dependencies (entries with Admin0 and Admin1).
  *	Demark, France, Netherlands: mother countries and their dependencies have different code3, therefore UID = code 3. (e.g., Faroe Islands, Denmark, UID = code3 = 234; Denmark UID = 208)
  *	United Kingdom: the mother country and dependencies have different code3s, therefore UID = code 3. One exception: Channel Islands is using the same code3 as the mother country (826), and its artificial UID = 8261.
  *	Australia: alphabetically ordered all states, and their UIDs are from 3601 to 3608. Australia itself is 36.
  *	Canada: alphabetically ordered all provinces (including cruise ships and recovered entry), and their UIDs are from 12401 to 12415. Canada itself is 124.
  *	China: alphabetically ordered all provinces, and their UIDs are from 15601 to 15631. China itself is 156. Hong Kong, Macau and Taiwan have their own code3.
  *	Germany: alphabetically ordered all admin1 regions (including Unknown), and their UIDs are from 27601 to 27617. Germany itself is 276.
  * Italy: UIDs are combined country code (380) with `codice_regione`, which is from [Dati COVID-19 Italia](https://github.com/pcm-dpc/COVID-19). Exceptions: P.A. Bolzano is 38041 and P.A. Trento is 38042.
3.	The US (most entries with Admin0, Admin1 and Admin2).
  *	US by itself is 840 (UID = code3).
  *	US dependencies, American Samoa, Guam, Northern Mariana Islands, Virgin Islands and Puerto Rico, UID = code3. Their Admin0 FIPS codes are different from code3.
  *	US states: UID = 840 (country code3) + 000XX (state FIPS code). Ranging from 8400001 to 84000056.
  *	Out of [State], US: UID = 840 (country code3) + 800XX (state FIPS code). Ranging from 8408001 to 84080056.
  *	Unassigned, US: UID = 840 (country code3) + 900XX (state FIPS code). Ranging from 8409001 to 84090056.
  *	US counties: UID = 840 (country code3) + XXXXX (5-digit FIPS code).
  *	Exception type 1, such as recovered and Kansas City, ranging from 8407001 to 8407999.
  *	Exception type 2, only the New York City, which is replacing New York County and its FIPS code.
  *	Exception type 3, Diamond Princess, US: 84088888; Grand Princess, US: 84099999.
  * Exception type 4, municipalities in Puerto Rico are regarded as counties with FIPS codes. The FIPS code for the unassigned category is defined as 72999.
4. Population data sources.
 * United Nations, Department of Economic and Social Affairs, Population Division (2019). World Population Prospects 2019, Online Edition. Rev. 1. https://population.un.org/wpp/Download/Standard/Population/
 * eurostat: https://ec.europa.eu/eurostat/web/products-datasets/product?code=tgs00096
 * The U.S. Census Bureau: https://www.census.gov/data/datasets/time-series/demo/popest/2010s-counties-total.html
 * Mexico population 2020 projection: [Proyecciones de población](http://sniiv.conavi.gob.mx/(X(1)S(kqitzysod5qf1g00jwueeklj))/demanda/poblacion_proyecciones.aspx?AspxAutoDetectCookieSupport=1)
* Brazil 2019 projection: ftp://ftp.ibge.gov.br/Estimativas_de_Populacao/Estimativas_2019/
* Peru 2020 projection: https://www.citypopulation.de/en/peru/cities/
* India 2019 population: http://statisticstimes.com/demographics/population-of-indian-states.php
* The Admin0 level population could be different from the sum of Admin1 level population since they may be from different sources.

Disclaimer: \*The names of locations included on the Website correspond with the official designations used by the U.S. Department of State. The presentation of material therein does not imply the expression of any opinion whatsoever on the part of JHU concerning the legal status of any country, area or territory or of its authorities. The depiction and use of boundaries, geographic names and related data shown on maps and included in lists, tables, documents, and databases on this website are not warranted to be error free nor do they necessarily imply official endorsement or acceptance by JHU.
