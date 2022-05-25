# Install
For installation and basic usage instructions, please refer to [fast-reid](https://github.com/JDAI-CV/fast-reid)
# Dataset
market1501 [GoogleDriver](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view)
<!-- [dukeMTMC](https://github.com/sxzrt/DukeMTMC-reID_evaluation) -->
dukeMTMC [GoogleDriver](https://drive.google.com/open?id=1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O)
MSMT17 [GoogleDriver](https://drive.google.com/file/d/1X12SuDvZlSr6V9gn9TXw2A7R1BYRblUH/view?usp=sharing)
# Evaluation
market1501
```bash
python tools/train_net.py --config-file fastreid/configs/osnet/dmosnet_market1501.yml  --eval-only MODEL.WEIGHTS fastreid/saveweights/model_best_zl_osnet100.pth  MODEL.DEVICE "cuda:0
```
dukeMTMC
```bash
python tools/train_net.py --config-file fastreid/configs/osnet/dmosnet_duke.yml  --eval-only MODEL.WEIGHTS fastreid/saveweights/model_best_dmosent100_duke.pth  MODEL.DEVICE "cuda:0"
```
MSMT17
```bash
python tools/train_net.py --config-file fastreid/configs/osnet/dmosnet_msmt.yml  --eval-only MODEL.WEIGHTS fastreid/saveweights/model_best_dmosent100_msmt.pth  MODEL.DEVICE "cuda:0"
```

# DM-OSNet configuration files and weights
model|config|weight
|:---|:---|:---
|market1501|[dmosnet_market1501.yml](fastreid/configs/osnet/dmosnet_market1501.yml )|[model_best_dmosnet100_market.pth](fastreid/saveweights/model_best_dmosnet100_market.pth )
|dukeMTMC|[dmosnet_duke.yml](fastreid/configs/osnet/dmosnet_duke.yml)|[model_best_dmosent100_duke.pth](fastreid/saveweights/model_best_dmosent100_duke.pth )
|MSMT17|[dmosnet_msmt.yml](fastreid/configs/osnet/dmosnet_msmt.yml)|[model_best_dmosent100_msmt.pth](fastreid/saveweights/model_best_dmosent100_msmt.pth )
