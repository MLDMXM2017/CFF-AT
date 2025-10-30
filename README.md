## CFF-AT
Data, source code, supplementary file of "Multi-task Cascade Forest Framework for Predicting Acute Toxicity Across Species".

## Data Source
(1)All toxicology datasets used in our study are available on the published data platform TOXRIC at https://toxric.bioinforai.tech/home.<br>
(2)The feature folder contains 59 toxicity datasets, the feature type is Avalon.<br>
(3)The label folder contains 59 label files corresponding to 59 toxicity datasets.<br>
(4)The label is in the last column of the label file.<br>
(5)Data description refers to the section A of Supplementary Material.<br>

Data Reference<br>
[1] Jain S, Siramshetty VB, Alves VM, Muratov EN, Kleinstreuer N, Tropsha A, Nicklaus MC, Simeonov A, Zakharov AV. Large-scale modeling of multispecies acute toxicity end points using consensus of multitask deep learning methods. Journal of Chemical Information and Modeling. 2021;61(2):653–663. https://doi.org/10.1021/acs.jcim.0c01164.<br>
[2] Wu LL, Yan BW, Han JS, Li RJ, Xiao J, He S, Bo XC. Toxric: A comprehensive database of toxicological data and benchmarks. Nucleic Acids Research. 2023. https://doi.org/10.1093/nar/gkac1074.

## Requirements
The running environment for Deep Forest and ADAPT needs to be installed.<br>
Please refer to the following website for the installation steps.<br>
https://deep-forest.readthedocs.io/en/latest/installation_guide.html <br>
https://adapt-python.github.io/adapt/install.html <br>

Note:<br>
(1)The method proposed in this study is an improvement based on Deep Forest framework.<br>
(2)ADAPT is a Python package providing some well known domain adaptation methods.<br>
(3)Sometimes, the installation environments of Deep Forest and ADAPT may cause conflicts, depending on the software upgrades of the corresponding communities. It is possible to consider calculate the distance relationship among toxicity endpoints in ADAPT environment in advance, and then implement the proposed method in Deep Forest environment.

## Code Description
(1)"CFFAT_5FoldCV.py" demonstrates the training and testing process of conducting a 5-fold cross-validation on 59 toxicity endpoints.<br>
(2)"CFFAT_5FoldCV_Result_Demo.log" presents the detailed running results of the code.



