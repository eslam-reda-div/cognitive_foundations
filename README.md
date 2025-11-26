<div align="center">

<p align="center"><img src="https://github.com/pkargupta/cognitive_foundations/blob/main/figs/readme_image.png" alt="Cognitive Foundations"/></p>

[![Static Badge](https://img.shields.io/badge/Paper-white?style=for-the-badge&logo=arxiv&logoColor=%23e46e2f&color=%232e4969)](https://arxiv.org/abs/2511.16660)
[![Static Badge](https://img.shields.io/badge/Blog-white?style=for-the-badge&logo=notion&logoColor=%23e46e2f&color=%232e4969)](https://tinyurl.com/cognitive-foundations)
[![Static Badge](https://img.shields.io/badge/Dataset-white?style=for-the-badge&logo=huggingface&logoColor=%23e46e2f&color=%232e4969)](https://huggingface.co/collections/stellalisy/cognitive-foundations)

[Priyanka Kargupta*](https://pkargupta.github.io/), [Shuyue Stella Li*](https://stellalisy.com/), [Haocheng Wang](https://hassonlab.princeton.edu/publications/contributor/wang-haocheng), [Jinu Lee](https://jinulee-v.github.io/), [Shan Chen](https://shanchen.dev/), [Orevaoghene Ahia](https://orevaahia.github.io/), [Dean Light](https://www.linkedin.com/in/dean-light), [Thomas L. Griffiths](https://cocosci.princeton.edu/tom/index.php), [Max Kleiman-Weiner](https://faculty.washington.edu/maxkw/), [Jiawei Han](https://hanj.cs.illinois.edu/), [Asli Celikyilmaz](http://asli.us/), [Yulia Tsvetkov](https://homes.cs.washington.edu/~yuliats/)

_*Equal contribution in alphabetical order_

</div>

# Cognitive Foundations for Reasoning and Their Manifestation in LLMs

## Links

- [Overview](#overview)
  - [Installation](#installation)
- [Assessing Behavioral Manifestation of Cognitive Elements](#assessing-behavioral-manifestation-of-cognitive-elements)
  - [Output Data Format](#output-data-format)
- [Test-Time Reasoning Guidance](#test-time-reasoning-guidance)
  - [Generating Guidance Templates](#generating-guidance-templates)
- [Citations](#citation)


## Overview

Our framework bridges **cognitive science** and **large language model (LLM) research** to systematically understand how LLMs reason and to diagnose/improve their reasoning processes, based on analysis of 192K model traces and 54 human think-aloud traces.

### Installation
The code is written in Python 3.10.9. The Python dependencies are summarized in the file `requirements.txt`. You can install them like this:
```
pip install -r requirements.txt
```

## Assessing Behavioral Manifestation of Cognitive Elements

We develop a taxonomy of **28 cognitive elements** spanning reasoning goals & properties, meta-cognitive controls, reasoning & knowledge representations, and transformation operations, creating a shared vocabulary between cognitive science and LLM research. We utilize this framework to encode reasoning traces into a **heterogenous graph**, where each node represents a cognitive element and edges between them reflect their temporal and hierarchical relationships.

<p align="center"><img src="https://github.com/pkargupta/cognitive_foundations/blob/main/figs/taxonomy.png" alt="Cognitive Foundations"/></p>

Our evaluation encompasses **192K+ model traces** from **18 different LLMs** across text, vision, and audio modalities, alongside **54 human think-aloud traces** to enable direct comparison between human and machine reasoning patterns. We study both _well-structured_ (e.g., Algorithmic) to _ill-structured_ (e.g., Dilemma) problem types. We provide all span-level annotation prompts in `element_annotation`.

### Output Data Format

In order to run [test-time reasoning guidance](#test-time-reasoning-guidance), we expect the following JSON file format for each model's span-level annotation result. We automatically read all model-specific JSON files from a specified directory:

```
# One file per model
{
    "[question_id]_[model_name]": {
        "sample_id": "[question_id]_[model_name]",
        "question_id": [int: question_id],
        "task": [str: task],
        "model_name": [str: the name of the model],
        "problem_type": [either a string label of the problem type or a list of index ids (we will take the mode of the latter)],
        "correctness": [bool: whether the model's final answer is correct or incorrect],
        "element_annotation": {
            "[element_label]": {
                "score": [int: 0-2, where 0 indicates no element present, 1 for partially present, and 2 for strongly present],
                "spans": [list: each item is a list of length 2, indicating both the start and end span index]
            },
            ...
        }
    }
}
```

## Test-Time Reasoning Guidance

We introduce **test-time reasoning guidance** as a targeted intervention to explicitly scaffold cognitive patterns predictive of reasoning success. In greedy fashion, we determine the most success-prone reasoning structure (subgraph) for each problem type, based on our empirical analysis. We convert each into a prompt which guides a model's reasoning process, improving performance by up to 26.7% on ill-structured problems while maintaining baseline performance on well-structured ones.

### Generating Guidance Templates

To generate test-time reasoning guidance templates for different problem types, run the `construct_graphs.py` script:

```bash
python construct_graphs.py \
    --element_dir /path/to/span_annotations \
    --prompt_template_dir structure_guidance/prompt_templates \
    --output_dir reasoning_structure/output_consensus_graphs \
    --path_to_question_info /path/to/question_info.json \
    --max_nodes 7 \
    --overlap_threshold 0.8 \
    --parallel_threshold 20
```

**Arguments:**
- `--element_dir`: Directory containing span-level annotation files (in the format described above)
- `--prompt_template_dir`: Output directory for generated prompts (default: `structure_guidance/prompt_templates`)
- `--output_dir`: Output directory for consensus graph visualizations (default: `reasoning_structure/output_consensus_graphs`)
- `--path_to_question_info`: Path to JSON file containing question metadata
- `--max_nodes`: Maximum number of nodes in the consensus graph (default: 7)
- `--overlap_threshold`: Overlap threshold for span tree construction (default: 0.8)
- `--parallel_threshold`: Parallel threshold for span tree construction (default: 20)
- `--target_type`: Optional filter for specific problem type (default: processes all types)

**Output:**

The script generates prompts in `structure_guidance/prompt_templates` that can be input into any model (we used **Claude Sonnet 4.5**) to produce reasoning guidance templates. These templates are then used during test-time to scaffold the model's reasoning process.

**Example Resources:**
- Graph visualizations for each problem type (max_nodes=7): `reasoning_structure/output_consensus_graphs/7/`
- Generated prompts for constructing guidance templates: `structure_guidance/prompt_templates/`
- Final test-time guidance templates: `structure_guidance/guidance_templates/`

## Citation

```bibtex
@article{kargupta2025cognitive,
  title={Cognitive Foundations for Reasoning and Their Manifestation in LLMs},
  author={Kargupta, Priyanka and Li, Shuyue Stella and Wang, Haocheng and Lee, Jinu and Chen, Shan and Ahia, Orevaoghene and Light, Dean and Griffiths, Thomas L and Kleiman-Weiner, Max and Han, Jiawei and Celikyilmaz, Asli and Tsvetkov, Yulia},
  journal={arXiv preprint arXiv:2511.16660},
  year={2025}
}
```