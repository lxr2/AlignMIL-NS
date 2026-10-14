# Aligning Prototypes and Updating Null Spaces for Continual WSI Learning

Official code for the MICCAI 2026 paper [Aligning Prototypes and Updating Null Spaces for Continual WSI Learning](https://papers.miccai.org/miccai-2026/0045-Paper2958.html).

The continual learning framework follows [Mammoth](https://github.com/aimagelab/mammoth), and the implementation builds on [QPMIL-VL](https://github.com/can-can-ya/QPMIL-VL).

## Installation

Install [uv](https://docs.astral.sh/uv/), then create the environment from the lock file:

```bash
uv sync
```

The locked PyTorch build targets CUDA 13. If your driver is older, point `torch` to a matching index in `pyproject.toml` (see the [uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/)) and re-run `uv sync`.

Download the [CONCH](https://huggingface.co/MahmoodLab/CONCH) weights and put them in `checkpoints/conch/pytorch_model.bin`.

## Data

The split and label files for the five datasets are in `data/`. You need to provide your own patch features. Following the paper, extract 256x256 patches at 20x with [CLAM](https://github.com/mahmoodlab/CLAM) and encode them with CONCH:

```bash
# in the CLAM repo
python create_patches_fp.py --source <wsi_dir> --save_dir <patch_dir> --patch_size 256 --seg --patch --stitch
CONCH_CKPT_PATH=<conch_ckpt> python extract_features_fp.py --data_h5_dir <patch_dir> --data_slide_dir <wsi_dir> \
    --csv_path <patch_dir>/process_list_autogen.csv --feat_dir <feat_dir> --batch_size 512 --slide_ext .svs --model_name conch_v1
```

Choose `--patch_level` so that patches are at 20x. Put the resulting `pt_files/` (one file per slide, named by its `pathology_id`) in the layout below:

```
data/
├── camelyon16/
│   ├── datasplit/fold_{1..5}.npz
│   ├── table/CAMELYON16_ic_cases.csv
│   └── feat_mag20_ps256_CONCH/pt_files/<pathology_id>.pt   # [N, 512]
├── tcga_lung/
├── tcga_brca/
├── tcga_esca/
└── tcga_rcc/
```

## Training

To run all 5 folds in both task orders:

```bash
GPU=0 bash scripts/train.sh
```

To run a single fold:

```bash
# forward order: CAMELYON16 -> LUNG -> BRCA -> ESCA -> RCC
uv run python main.py -f configs/ex_cl_fw.yaml seed=1 name=alignmil
# reverse order
uv run python main.py -f configs/ex_cl_rv.yaml seed=1 name=alignmil
```

The default configs are the ones used for the reported results. A few implementation details differ from the paper text: the learning rate is 0.01; the class similarity loss of QPMIL-VL (`beta=0.5`) is added to the total loss; the null space is spanned by up to the 20 smallest singular values; and the null-space update (Eq. 8) is applied after every epoch of later tasks with step size `tunable_v_lr`.

Any config key can be overridden on the command line (e.g. `cuda_id=1 wandb=online`). Results are saved to `results/<config>/<name>/train-data_split_seed_<seed>/`. After the `seed=5` run, the results of all five folds are summarized in `results/<config>/<name>/metrics/`.

## Acknowledgements

[Mammoth](https://github.com/aimagelab/mammoth), [QPMIL-VL](https://github.com/can-can-ya/QPMIL-VL), [CONCH](https://github.com/mahmoodlab/CONCH), [CLAM](https://github.com/mahmoodlab/CLAM), [L2P](https://github.com/JH-LEE-KR/l2p-pytorch), [TOP](https://github.com/miccaiif/TOP), [TaskRes](https://github.com/geekyutao/TaskRes).

## Citation

```bibtex
@InProceedings{LiXia_Aligning_MICCAI2026,
  author    = {Li, Xianrui and Xiao, Kaiwen and Chan, Antoni B.},
  title     = {Aligning Prototypes and Updating Null Spaces for Continual WSI Learning},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
  year      = {2026},
  publisher = {Springer Nature Switzerland},
  volume    = {LNCS 16891}
}
```

## License

This repository is released under the [CC BY-NC-ND 4.0](LICENSE) license and may only be used for non-commercial, academic research purposes with proper attribution. It contains code adapted from [QPMIL-VL](https://github.com/can-can-ya/QPMIL-VL) and [CONCH](https://github.com/mahmoodlab/CONCH) (`models/conch/`), both released under CC BY-NC-ND 4.0. The CONCH model weights are subject to their own [terms of use](https://huggingface.co/MahmoodLab/CONCH).
