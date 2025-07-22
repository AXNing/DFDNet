# DFDNet: Dynamic Frequency-Guided De-Flare Network





### Update

- **2024.12.29**: This repo is created.

### Installation

1. Clone the repo

    ```bash
    git clone https://github.com/AXNing/DGDNet.git
    ```

1. Install dependent packages

    ```bash
    cd DGDNet
    pip install -r requirements.txt
    ```




### Data Download

Download the Flare7k++ dataset [here](https://github.com/ykdai/Flare7K).

#### Flare7k++ structure

```
├── Flare7K
    ├── Reflective_Flare 
    ├── Scattering_Flare
         ├── Compound_Flare
         ├── Glare_with_shimmer
         ├── Core
         ├── Light_Source
         ├── Streak
├── Flare-R
	├── Compound_Flare
	├── Light_Source
├── test_data
     ├── real
          ├── input
          ├── gt
          ├── mask
     ├── synthetic
          ├── input
          ├── gt
	  ├── mask

```



### Training model


**Training**

Please use:

```
python basicsr/train.py -opt options/DGDNet_option.yml
```
To train a model with your own data/model, you can edit the `options/DGDNet_option.yml` 



### Inference Code
To estimate the flare-free images with the checkpoint pretrained on Flare7K++ dataset, you can run the `test.sh` by using:

```
CUDA_VISIBLE_DEVICES=1    python test.py --input ./datasets/Flare7Kpp/test_data/real/input --output ./results  --model_type DGDNet --model_path ./.pth --flare7kpp
CUDA_VISIBLE_DEVICES=1   python evaluate.py --input ./results --gt ./datasets/Flare7Kpp/test_data/real/gt --mask ./datasets/Flare7Kpp/test_data/real/mask
```

### Test datasets
Flare7k++ real and synthetic nighttime flare-corrupted: [link](https://github.com/ykdai/Flare7K). 

Real-world nighttime flare-corrupted dataset: [link](https://github.com/ykdai/Flare7K).

Consumer electronics test datasets: [link](https://drive.google.com/drive/folders/1J1fw1BggOP-L1zxF7NV0pYhvuZQsmiWY).





### License

This project is licensed under <a rel="license" href="https://github.com/ykdai/Flare7K/blob/main/LICENSE">S-Lab License 1.0</a>. Redistribution and use of the dataset and code for non-commercial purposes should follow this license.

