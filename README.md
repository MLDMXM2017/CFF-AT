## CFF-AT
the source code, supplementary file of "Multi-task Cascade Forest Framework for Predicting Acute Toxicity Across Species".

## Requirements
Python==3.7.6    
numpy 
pandas      
sklearn    

## CodeAndData
(1)The feature folder contains 59 toxicity datasets, the feature type is Avalon. <br/>
(2)The label folder contains 59 label files corresponding to 59 toxicity datasets. <br/>
(3)The label is in the last column of the label file. <br/>
(4)"task_relation_cov_distance.csv", this csv file contains pre-calculated covariance relationship among subtasks.<br/>

## GetResults
#Run the following commands in linux environment to obtain 5 reports corresponding to 5-fold cross-verification
nohup python ./CFFAT_fold1.py > ./Fold1Result.log 2>&1 & <br/>
nohup python ./CFFAT_fold2.py > ./Fold2Result.log 2>&1 & <br/>
nohup python ./CFFAT_fold3.py > ./Fold3Result.log 2>&1 & <br/>
nohup python ./CFFAT_fold4.py > ./Fold4Result.log 2>&1 & <br/>
nohup python ./CFFAT_fold5.py > ./Fold5Result.log 2>&1 & <br/>

