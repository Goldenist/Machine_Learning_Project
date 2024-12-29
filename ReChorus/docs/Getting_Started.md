## Getting Started

1. Install [Anaconda](https://docs.conda.io/en/latest/miniconda.html) with Python >= 3.10
2. Clone the repository

```bash
git clone https://github.com/THUwangcy/ReChorus.git
```

3. Install requirements and step into the `src` folder

```bash
cd ReChorus
pip install -r requirements.txt
cd src
```

4. Run model with the build-in dataset

```bash
python main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-06 --dataset Grocery_and_Gourmet_Food
python main.py --model_name LightGCN --emb_size 64 --lr 1e-3 --l2 1e-06 --dataset Grocery_and_Gourmet_Food
python .\src\main.py --model_name JMP_GCF_k_012 --emb_size 64 --lr 1e-3 --l2 1e-06  --dataset Grocery_and_Gourmet_Food --gpu "0" --num_workers 0
python .\src\main.py --model_name JMP_GCF --emb_size 64 --lr 1e-3 --l2 1e-06  --dataset Grocery_and_Gourmet_Food --gpu "0" --num_workers 0
python .\src\main.py --model_name JMP_GCF --emb_size 64 --lr 1e-3 --l2 1e-06  --dataset MIND_Large\MINDTOPK --gpu "0" --num_workers 0
python .\src\main.py --model_name JMP_GCF --emb_size 64 --lr 1e-3 --l2 1e-06  --dataset MovieLens_1M\ML_1MTOPK --gpu "0" --num_workers 0
```

5. (optional) Run jupyter notebook in `dataset` folder to download and build new datasets, or prepare your own datasets according to [Guideline](https://github.com/THUwangcy/ReChorus/tree/master/data/README.md) in `data`

6. (optional) Implement your own models according to [Guideline](https://github.com/THUwangcy/ReChorus/tree/master/src/README.md) in `src`
