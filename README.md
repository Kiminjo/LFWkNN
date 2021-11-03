# LFWkNN

Local Feature Weight kNN combined Local kNN and Feature weighted kNN. This project was carried out as part of the project of the Advanced Machine Learning class in the Department of Data Science at Seoul National University of Science and Technology.
<br></br>

## Local Feature Weight kNN 

In this project, I propose a kNN model that is stronger for imbalanced data. The kNN algorithm is a method of classifying target data by referring to k nearest data labels that are geographically close to the target data. However, this method has a disadvantage in that it cannot guarantee high accuracy in the case of an unbalanced data set.

There have been many models for strengthening the kNN algorithm in the past. Representatively, there are several algorithms such as feature weight kNN and DCT kNN. 

In this study, we propose a kNN algorithm that is strong against imbalanced data by combining the ideas of the above-mentioned algorithms.

Please check [here](https://github.com/Kiminjo/data-mining-lecture/files/7465261/A.proposal.for.local.k.values.for.k-nearest.neighbor.rule.pdf) if you interested in Local kNN.

If you interested in Feature weighted kNN, please check [here](https://github.com/Kiminjo/data-mining-lecture/files/7465265/An.improved.kNN.based.on.class.contribution.and.feature.weighting.pdf)


<br></br>

## Dataset

In this experiment, a dataset frequently used in existing machine learning was used. One imbalanced dataset and two balanced datasets were used to compare the performance of the algorithm.

The detailed performance of the dataset is specified in the table below.
|Dataset|Feature number|Instancenumber|Class number|Proportion|Imbalanced status|
|:---:|:---:|:---:|:---:|:---:|:---:|
|Iris|4|150|3|50:50:50|X|
|Liver|6|340|2|195:145|X|
|Blood|4|740|2|562:178|O|

<br></br>

## Software Requirements

- python >= 3.5
- numpy 
- scikit-learn
- scipy

<br></br>

## Key Files 

- `local_k.py` : It contains code that implements local kNN.
- `feature_weight.py` : Implemented code related to feature weights and class contributions.
- `LFWkNN.py` : Main file of this project. Combine two method implemented in 'local_k.py' and 'feature_weight.py'
