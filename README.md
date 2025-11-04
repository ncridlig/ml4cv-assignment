# Semantic Segmentation of Unexpected Objects on Roads

## Run locally
This project is developed for CUDA v.12.9 within Linux. Install python 3.12 and then the dependencies.
```
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```
Download the necessary datasets.
```
chmod +x download.sh
./download.sh
```
To verify the datasets are loaded, you can run the dataloader utility.
```
python3 dataloader.py
```
Train samples: 8125
Validation samples: 4187
Test samples: 3000
Batch shape: torch.Size([32, 3, 224, 224])