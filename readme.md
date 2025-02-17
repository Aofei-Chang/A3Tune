## A³TUNE Directory Structure

The current directory structure is as follows:

```
A3Tune/
├── readme.md
├── LVLMs/
├── Segment/
```

### Weak Label Generation

The weak label generation process is located in the `Segment/` directory. Here, we generate weak labels for each dataset, which are then used for downstream fine-tuning.

### Main Experiments for A³TUNE

Our primary experiments for A³TUNE are located in the `LVLMs/llava-med/` directory.

- **Training Code**:
   The training code for `llava-med` can be found at:

  ```
  llava-med/llava/train/train.py
  ```

  In this file:

  - We import MoE module designs for **A³MoE** and incorporate additional parameters for training.

  - The preprocessing of weak labels is implemented in the `LazySupervisedDataset` class.

  - Implementation details are included in the forward function of the `LlavaLlamaForCausalLM` class.

    - The attention tuning loss function, `calculate_top_attention_loss` is defined in:

      ```
      LVLMs/llava-med/llava/model/utils.py
      ```

- **Training Scripts**:
   The training scripts for **A³TUNE** are located in:

  ```
  llava-med/scripts/train/moe/top_heads
  ```

### Inference Implementation

The inference implementation for both **A³TUNE** and baseline models can be found in the following directory:

```
llava-med/llava/eval/
```

- **Baselines**:
   The required baseline files are stored in:

  ```
  llava-med/llava/eval/
  ```

  This includes the baselines **avisc, PAI, and VCD**. Additionally, we integrate **DAMRO** and **M3ID** within the avisc paradigm.

- **Main Inference File**:
   The main file handling inference for both **A³TUNE** and baselines is:

  ```
  llava-med/llava/eval/model_vqa_med.py
  ```