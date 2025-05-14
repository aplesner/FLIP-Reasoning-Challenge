# FLIP Reasoning Challenge

This repository contains the code and dataset for the paper "FLIP Reasoning Challenge," which introduces a benchmark for evaluating AI reasoning capabilities based on human verification tasks from the Idena blockchain.

## Paper

Link to the paper: [FLIP Reasoning Challenge](https://arxiv.org/abs/2504.12256)

## Introduction

FLIP challenges present users with two orderings (stacks) of 4 images, requiring them to identify which ordering forms a coherent story. These tasks are designed to test complex reasoning abilities rather than simple recognition tasks, emphasizing sequential reasoning, visual storytelling, and common sense understanding.

Key features of the FLIP benchmark:
- Created from human-generated and human-verified tasks from the Idena blockchain
- Tests sequential reasoning and visual storytelling abilities
- Provides clear ground truth, making it easy to diagnose model failures
- High human performance baseline (95.3% accuracy)
- Current state-of-the-art models achieve maximum accuracies of 75.5% (open-source) and 77.9% (closed-source)

## Data

The FLIP dataset is available on Hugging Face:
[https://huggingface.co/datasets/aplesner-eth/FLIP-Challenge](https://huggingface.co/datasets/aplesner-eth/FLIP-Challenge)

### Dataset Statistics
- Total flips: 11,674
- Train set: 3,502 flips (30%)
- Validation set: 3,502 flips (30%)
- Test set: 4,670 flips (40%)
- Small subsets are also available for computationally intensive experimentation

Solutions are nearly evenly distributed between Left (49.4%) and Right (50.6%), with most challenges having strong consensus (95.7%).

### Task Format

Each task is stored as a JSON file with the following structure:

```json
{
    "task_id": "_flip_bafkreianuvtem5nababzw5z4iscr5ocvgaviilmemwn3o73jkak7bqrjde",
    "images": {
        "0": "46efd91c-be17-42b8-8f5e-2a84b96d21af",
        "1": "9d1fac84-0c9f-4ab7-9d3b-a3b4c61dc390",
        "2": "ceecdc8b-840c-46d7-b694-74f05839447f",
        "3": "cbdf27d1-aa84-405b-86db-cb336d0bc4a7"
    },
    "left_stack": ["2", "3", "1", "0"],
    "right_stack": ["3", "0", "2", "1"],
    "agreed_answer": ["Right", "Strong"],
    "votes": {
        "Left": "1", 
        "Right": "4", 
        "Reported": "0"
    },
    "details": {
        "Author:": "0x63f7aa6C19A0f7D4BBB4177000Af671ED212e490",
        "Epoch:": "#0027",
        "Size:": "86140 bytes",
        "Created:": "12/24/2019 13:23:51",
        "Block:": "669858",
        "Tx:": "0xdbca60c3d10770f4bc2f73fd9119d9509117a8db08196f128382bffbf3d8c79f"
    }
}
```

## Code Structure

The codebase is organized as follows:

- `src/` - Contains the main code for the experiments
  - `local_reasoning.ipynb` - Jupyter notebook for running and analyzing the models
  - `new_API_reason.py` - Clean implementation for running OpenAI models
  - Other utility scripts and modules for data processing and model evaluation

## Experimental Methodology

Our experiments evaluate various models on the FLIP dataset:

1. **Captioning Models**: Generate text descriptions of images
   - VIPLLAVA (7B & 13B)
   - LLAVANEXT (MISTRAL 7B, VICUNA 7B, VICUNA 13B)
   - BLIP2 (2.7B, 6.7B, FLAN T5 XXL)
   - LLAMA 3.2 (Vision - Instruct 11B)

2. **Open-Sourced Reasoning Models**: Process captions to determine the correct ordering
   - Llama 3.1 70B (Meta)
   - QWEN 2.5
   - QWEN 2 VL
   - Llama 3.1 Nemotron (70B & 51B from Nvidia)

3. **Closed-Sourced Reasoning Models**:
   - GPT-4 Turbo (OpenAI)
   - Gemini 1.5 Pro and Flash (Google)

4. **Additional Experiments**:
   - Prompt engineering and task reframing
   - Summarization of captions
   - Inference with historical context
   - Ensemble models

## Running the Code

The first version of the codebase was written by Turlan Kuzhagaliyev while working on his master's thesis. Andreas Plesner has updated `src/local_reasoning.ipynb` and created `src/new_API_reason.py` to provide a cleaner way to run the OpenAI models.

### Prerequisites

1. Download the dataset from Hugging Face:

2. Configure API keys for any closed-source models you want to use by creating a `.env` file:
   ```
   OPENAI_API_KEY=your_key_here
   GOOGLE_API_KEY=your_key_here
   ```

### Running Experiments

#### Using the Jupyter Notebook
1. Start Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open `src/local_reasoning.ipynb`
3. Follow the instructions in the notebook to run experiments with different models


## Key Results

Our experiments show:
- Best open-source models achieve 75.5% accuracy in zero-shot settings
- Best closed-source models reach 77.9% accuracy
- Human performance is 95.3% accurate
- Captioning models aid reasoning models by providing text descriptions
- Ensemble methods can boost performance to 85.2%

These findings highlight the gap between current AI capabilities and human-level reasoning on complex multimodal tasks.

## Citation

If you use this dataset or code in your research, please cite:

```
@inproceedings{plesner2025flip,
  title={FLIP Reasoning Challenge},
  author={Plesner, Andreas and Kuzhagaliyev, Turlan and Wattenhofer, Roger},
  booktitle={First Workshop on Open Science for Foundation Models at ICLR 2025},
  year={2025}
}
```

## Contact

For questions or feedback, please contact:
- Andreas Plesner (aplesner@ethz.ch)
