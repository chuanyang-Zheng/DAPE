# DAPE
The this is the official implementation of "DAPE: Data-Adaptive Positional Encoding for Length Extrapolation"

---
## 🚩 **New Features/Updates**
- ✅ Oct. 09, 2024. 💥 Please check our new paper [DAPE V2: Process Attention Score as Feature Map for Length Extrapolation](https://arxiv.org/abs/2410.04798). TL;DR: we identify and interpret the Transformer length extrapolation problem as a result of the limited expressiveness of the naive query and key dot product, and we successfully translate the length extrapolation issue into a well-understood feature map processing problem. 
- ✅ Oct. 06, 2024. 💥 We release all the code!
- ✅ Sep. 26, 2024. 💥 The paper is accepted to NeurIPS 2024!
- ✅ May. 23, 2024. 💥 We upload our implementation of CAPE. More code is coming soon.

---
**Installation**
---


First make sure you are in an environment with Python 3.8 with an appropriate version of PyTorch 1.8 or later installed. **Note:** Some of the libraries that GPT-NeoX depends on have not been updated to be compatible with Python 3.10+. Python 3.9 appears to work, but this codebase has been developed and tested for Python 3.8.

To install the remaining basic dependencies, run:

```bash
pip install -r requirements/requirements.txt
```
For details, please refer to GPT-NeoX README.

---
**Prepare Data**
---

As the EleutherAI delete the Pile dataset, we have to download the Pile-Arxiv and Pile-Books3 via Magnet URI scheme.

The Magnet URL scheme Link: magnet:?xt=urn:btih:0d366035664fdf51cfbe9f733953ba325776e667
```bash
Download Step:
1. Go to the websit https://webtor.io/#/
2. Copy and Paste the link   "magnet:?xt=urn:btih:0d366035664fdf51cfbe9f733953ba325776e667".
3. Choose the Dataset(such as Arxiv or Books3)
4. Download the dataset via weget.
```


After that, you could prepare the data via the command. 
```bash
python prepare_data.py -d ./data --dataset arxiv 
python prepare_data.py -d ./data --dataset books3 
```
For more details, please refer to the README_gpt_neox.md.


---
**Train**
---
Use the command
```bash
python ./deepy.py train.py -d configs 125M.yml local_setup.yml
```
Specificlly, try use the following commands to repdocue our results:
Use the command
```bash
#For NoPE
python ./deepy.py train.py -d configs 125M/512/125M_none.yml local_setup.yml

#For RoPE
python ./deepy.py train.py -d configs 125M/512/125M.yml local_setup.yml

#For T5's bias
python ./deepy.py train.py -d configs 125M/512/125M.yml local_setup.yml

#For CoPE
python ./deepy.py train.py -d configs 125M/512/125M_cope.yml local_setup.yml

#For Alibi
python ./deepy.py train.py -d configs 125M/512/125M_alibi.yml local_setup.yml

#For Kerple
python ./deepy.py train.py -d configs 125M/512/125M_kerple.yml local_setup.yml

#For DAPE-FIRE
python ./deepy.py train.py -d configs 125M/512/125M_fire.yml local_setup.yml

#For DAPE-Alibi
python ./deepy.py train.py -d configs 125M/512/125M_alibi_c.yml local_setup.yml

#For DAPE-Kerple
python ./deepy.py train.py -d configs 125M/512/125M_kerple_c.yml local_setup.yml

#For DAPE-FIRE
python ./deepy.py train.py -d configs 125M/512/125M_fire_c.yml local_setup.yml

#For DAPEV2: DAPE_{1x3}-Alibi
python ./deepy.py train.py -d configs 125M/512/125M_alibi_capev2.yml local_setup.yml

#For DAPEV2: DAPE_{1x3}-Kerple
python ./deepy.py train.py -d configs 125M/512/125M_kerple_capev2.yml local_setup.yml

#For DAPEV2: DAPE_{1x3}-FIRE
python ./deepy.py train.py -d configs 125M/512/125M_fire_capev2.yml local_setup.yml
```

---
**Evaluate**
---
The code will automatically evaluate the result every 5000 steps.
```bash
To reproduce the result of our paper, please set the "eval_iters"(in config yml files) to 20
```
---
**Furhter Question**
---
If there you have any questions, please ask a issue in this GitHub repository.
