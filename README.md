This code is for paper "Interpretable Disease Prediction via Path Reasoning over Medical Knowledge Graphs and Admission History".

The key files are:

- `train.py`: Used for training and validating the models. All parameters use default values, but can also be set at runtime.
- `metrics.py`: Contains all evaluation metrics used in the paper. 
- The `modeling/` folder contains the model implementations.
- The `util/` folder contains utility functions like data processing and transformations.
- The `data/` folder is meant for storing the data, but we cannot provide it due to restrictions. Please download it yourself following the instructions provided.

  You can download the MIMIC III from : https://physionet.org/content/mimiciii/1.4/

  MIMIC IV from: https://physionet.org/content/mimiciv/2.2/

To obtain the data, you need to register as a credentialed user on PhysioNet, complete the required CITI training on research ethics, and sign the PhysioNet Data Use Agreement. Please refer to the database overview pages for complete instructions.

To run:

```
python train.py
```

The code shows an example using the MIMIC III dataset, but can be adapted for MIMIC IV by changing the data paths.

- NOTE: Please download the MIMIC dataset yourself before running. Contributions and improvements are welcome!

#  Acknowledge 

In our work, part of the code is referenced from the following open-source code: **

1. QA-GNN: Question Answering using Language Models and Knowledge Graphs. https://github.com/michiyasunaga/qagnn

2. HiTANet: Hierarchical Time-Aware Attention Networks for Risk Prediction on Electronic Health Records. https://github.com/HiTANet2020/HiTANet

Many thanks to the authors and developers!
