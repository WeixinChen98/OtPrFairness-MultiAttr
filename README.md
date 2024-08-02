

# Investigating User-Side Fairness in Outcome and Process for Multi-Type Sensitive Attributes in Recommendations

This repository includes the implementation for paper *Investigating User-Side Fairness in Outcome and Process for Multi-Type Sensitive Attributes in Recommendations*.

## Datasets
The preprocessed Insurance and MovieLens-1M datasets are already provided in the `./dataset` folder. 

## Environments

The experimental environment is Python 3.8.5 with the following packages:
```
numpy==1.20.1
torch==1.8
pandas==1.2.4
scipy==1.6.2
tqdm==4.32.1
scikit_learn==0.23.1
```

Create and activate a new [Anaconda](https://www.anaconda.com/) environment with Python 3.8.5:
```
> conda create -n fair_recommendation python=3.8.5
> conda activate fair_recommendation
```

Install the above packages:
```
> pip install -r ./src/requirements.txt
```

## Usage
All running commands can be found in the `./cmd` folder. For example, to train the  PCFR fairness method with PMF as recommendation benchmark for gender as sensitive attribute on the Insurance dataset:
```
> cd ./src/
> python ./main.py --model_name PMF --fairness_framework PCFR --optimizer Adam --dataset insurance --feature_columns u_gender --data_processor RecDataset --metric ndcg@3,f1@3 --l2 1e-4 --lr 1e-3 --batch_size 1024 --model_path "../model/PMF_PCFR_insurance_u_gender_neg_sample=10/PMF_PCFR_insurance_u_gender_l2=1e-4_dim=64_neg_sample=10.pt" --runner RecRunner --vt_num_neg 10 --vt_batch_size 1024 --num_worker 0 --epoch 1000 --eval_disc 
```



## Acknowledgement
The code of this repository is implemented based on the source code framework at https://github.com/yunqi-li/Personalized-Counterfactual-Fairness-in-Recommendation.