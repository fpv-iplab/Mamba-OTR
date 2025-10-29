# Mamba-OTR: a Mamba-based Solution for Online Take and Release Detection from Untrimmed Egocentric Video

This work hosts the code related to the following paper:

- Online Detection of End of Take and Release Actions from Egocentric Videos
- Mamba-OTR: a Mamba-based Solution for Online Take and Release Detection from Untrimmed Egocentric Video

## Overview

This repository provides the following components:

1. The official PyTorch implementation of the proposed Mamba-OTR model;
2. A program to train, validate and test the proposed method on the EPIC-KITCHENS-100 datasets;
3. The checkpoints of the models trained on the EPIC-KITCHENS-100 datasets;

Please, refer to the paper for more technical details. The following sections document the released material.


## Environment

*   **Python:** 3.11.9
*   **Conda:** Recommended for managing dependencies.
    ```bash
    conda create -n mamba_otr python=3.11.9
    conda activate mamba_otr
    ```
*   **Requirements:** Install dependencies using `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```


## Usage

### Training

To train the Mamba-OTR model, use the `train_net.py` script:

```bash
python train_net.py --config_file <path_to_config_file> --gpu <gpu_id> 
```

### Inference

To run inference and evaluate the model, use the `test_net.py` script:

```bash
python test_net.py --config_file <path_to_config_file> --gpu <gpu_id> MODEL.CHECKPOINT <path_to_checkpoint> 
```


## Citations

If you use this code or model, please cite:
```
@inproceedings{catinello2025online,
  author = { Alessandro Sebastiano Catinello and Giovanni Maria Farinella and Antonino Furnari },
  title = {  Online Detection of End of Take and Release actions from Egocentric Videos },
  booktitle = {  International Conference on Computer Vision Theory and Applications (VISAPP)   },
  year = { 2025 },
}

@inproceedings{Catinello2025MambaOTR,
  year = { 2025 },
  booktitle = { Proceedings of the 23rd International Conference on Image Analysis and Processing (ICIAP) },
  title = { Mamba-OTR: a Mamba-based Solution for Online Take and Release Detection from Untrimmed Egocentric Video },
  author = { Alessandro Sebastiano Catinello and Giovanni Maria Farinella and Antonino Furnari },
}

```


## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
