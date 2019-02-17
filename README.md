# Predict-Market-Competitiveness-For-Insurance-Products
This repository contains the approach that led us to win the MLDS Republic Day <a href="https://www.machinehack.com/course/offline-hackathon-republic-day-hackathon-by-teg-analytics-and-aim/">Hackathon</a>. This <a href="https://www.analyticsindiamag.com/how-these-data-science-enthusiasts-from-christ-university-solved-our-insurance-products-hackathon/">link</a> contains the article featuring us in the Analytics India Magazine.<br>

This data science hackathon was conducted by TEG Analytics and Analytics India Magazine.The theme of the data science hackathon is “Predict Market Competitiveness For Insurance Products”. The description for the hackathon can be found <a href="https://github.com/SamdenLepcha/Predict-Market-Competitiveness-For-Insurance-Products/blob/master/Republic%20Day%20Hackathon.pdf">here</a>. <br>

<b>Below we describe two approaches to the problem we made:</b><br>

<b>1st. Time series GRU/LSTM:</b></br>

Initially after getting the extra data for sales of January 2017 we tried a time series approach to the problem with Keras and with intermediate GRU and LSTM layers with certain time based and lag features. But unfortunately, our predictions didn't seem to perform quite well on the leaderboard.<br>

Upon further EDA and probing we came to know that the sales trend from 2017-18 January was actually not followed in 2019 January. Moreover, the lack of adequate timesteps made the model believe the trend might follow which was actually not the case!! Thus, we shifted to primitive models like boosted trees and stacking a variety of models to obtain efficient results.<br><br>


<b>2nd. Stacked Approach</b>

After successful EDA and preprocessing we started our modelling. We used a stack approach of various models like Knn Regressor, Random Forest Regressor, GradientBoosting Regressor. AdaBoostRegressor and ExtraTrees Regressor. Intuition into the data compelled us to use stacked approach rather than plain ensembling of various models.<br>

Further we divided the data into a dataset with the same overlapping keys between the test and the train and a different set of keys between the test and the train. For the same keys, we trained it on the train and tested it on the test. For the different keys we took the features to the corresponding keys from the test, took the dependant variable from the crosswalks provided and made a train set. Thus, after that we fit it on the train and tested it on the test.<br>

After spending hours on the data, we deduced that the problem wanted us to overfit on the train. A good leaderboard score could only be obtained by overfitting as the ‘mean sales actually didn’t rise much in 2019 as compared to 2018’. Further timesteps would have probably helped the LSTM to train better but just 2 made the model think that the sales trend would actually continue, which was actually not the case!<br>

Our final submission files contains a model which a good balance between bias and variance. Proper data pre-processing helped us achieve so. Also attached, is a file where we showcase our basic EDA using seaborn and plotly. The advanced part of the EDA was done on Kaggle in plotly due to a lack of the desktop version of the plotly.<br>

<b>Now we will be taking about the preprocessing part of the data and how we went about it using various logical deductions based on concrete postulations.</b><br>

<b>SameKey<b>- Dataset which has the same set of combinations of County, State and bid_id in both the provided test and train.<br> 

<b>UniqueKey<b>- Dataset which has the combination of County, State and bid_id which are in the test but not in the train.<br>

1.	After merging our test data with the same keys or the unique keys we dropped the following features to create two new data frames (df1 and df2) from the test and train .<br>

•	We dropped "segment_id, "Plan_ID" and “Contract_ID” because they together make up the “bid_id” therefore they are not required.<br>
•	We dropped “statecode” because it is just a short form of “State”<br>
•	"FIPS_State_County_Code", "SSA_State_County_Code" were dropped because the code value corresponded with “State” and “County”.<br>
•	"SalesMarket" was dropped because this was the same as “County”.<br>
•	"National_PDP" was dropped because it contained all null values.<br>
•	“Snp” and “Special_Needs_Plan" were similar and hence the former was dropped.<br>
•	“TotalEstimatedAnnualCost” was dropped because this feature only contained null values and later on we created a similar feature to this.<br>
•	“Drug_Benefit_Type_Detail” was dropped because this feature and “Benefit_Type” were similar.<br>
•	'Crdiovsculr_Disordrs_Diabetes' was dropped because this feature only contained null values.<br>
•	"Parent_Organization " was dropped because this feature was equal to “Organization_Name” feature. The “Organization_Name” was later dropped because some of the Organization names changed from 2017 to 2018 and because no two organization names can have the same bid_id and hence wasn’t needed.<br>

2.	The nan values of “PartD” mapped to certain features like (“PartD_Basic_Premium3”, “PartD_Supplemental_Premium4”, “PartD_Total_Premium5”) had values to it which is only possible if the PartD of the insurance exists therefore all the nan values of the PartD were filled with “Yes”.<br>

3.	The features like “PartD_Basic_Premium3”, “PartC_Premium2”,” PartD_Supplemental_Premium4”,” PartD_Total_Premium5”, “ PartD_Prm_oblgtn_Full_prm_assist”, “PartD_Prm_oblgtn_75_prm_assist”, “PartD_Prm_oblgtn_50_prm_assist”, “PartD_Prm_oblgtn_25_prm_assist”, “PartD_Drug_Deductible” which had nan values were imputed with $0.00 were then converted to float using various functions and then the dollar sign was removed from them.<br>

4.	The “Overall_Star_Rating” and “star rating” were almost the same so we joined them and created new feature.<br>

5.	All the nan values of various health features like "Cardiovascular_Disorders", "Dementia", "Diabetes_Mellitus", "HIV_AIDS", "Chronic_Lung_Disorders", "Chrnic_Mental_hlth_Cdtn", "Crdiovsculr_Disordrs_hrt_fail", "Cardio_Disordrs_hrtfail_diabetes", "End_stge_Renal_Disease_dialysis", "Chronic_Heart_Failure", "Chronic_Heart_Fail_Diabetes"  were filled with 0.<br>

6.	The nan values of “Monthly_Consolidated_Premium_C_D” were filled with $0.00.<br>
7.	If MOOP is not nan then we keep MOOP as the same. Otherwise, If MOOP is nan and Add_Cvrg_Offrd_in_Gap is “Yes” then we fill it up with $0.00. Otherwise we fill it up with Annual_Drug_Deductible.<br>
8.	YearlyMin is the minimum expenditure an owner of the insurance scheme needs to pay. It is calculated by multiplying 12 with the monthly premium.<br>
9.	YearlyMax is the maximum expenditure an owner of the insurance scheme needs to pay. It is calculated by multiplying 12 with the monthly premium and adding the MOOP with it.<br>
10.	Additionally State, bid_id and Location (combination of State and County) were mean encoded in both the train and test. 
11.	Most of the benefits dataset were empty and even after adding some they didn’t add much value to the model. Tests were done on many of the features but unfortunately none of them seemed to improve the accuracy. <br>

All these features(specially the last 5) can be understood properly with the help of this following key:<br>

<b>Yearly deductible for drug plans</b>- This is the amount you must pay each year for your prescriptions before your Medicare drug plan pays its share.<br>

<b>Added coverage gap</b>- It is an add-on to existing plan which covers your MOOP.<br>

<b>MOOP</b>- Medicare Advantage plan’s MOOP or Maximum Out-of-Pocket limit is the total amount one needs to spend in a year on co-payments and co-insurance for covered or eligible medical services.  So, when one has reached his/her annual MOOP limit, his/her Medicare Advantage plan's eligible medical services are covered for the remainder of the year at no cost to him/her.<br>


![51047931_2151868531502920_5347948296063156224_o](https://user-images.githubusercontent.com/33536225/52916173-38f45a80-3302-11e9-9248-064fadef3876.jpg)
