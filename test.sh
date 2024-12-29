
CUDA_VISIBLE_DEVICES=1    python test.py --input ./datasets/Flare7Kpp/test_data/real/input --output ./results  --model_type DGDNet --model_path ./.pth --flare7kpp
CUDA_VISIBLE_DEVICES=1   python evaluate.py --input ./results --gt ./datasets/Flare7Kpp/test_data/real/gt --mask ./datasets/Flare7Kpp/test_data/real/mask





