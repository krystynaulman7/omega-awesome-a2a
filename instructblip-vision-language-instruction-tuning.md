# InstructBLIP: Towards General-Purpose Vision-Language Models with Instruction Tuning

**Type:** Vision-Language Model  
**Link:** [Paper](https://arxiv.org/abs/2305.06500) | [GitHub](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip)

## Analysis
InstructBLIP represents a breakthrough in vision-language instruction tuning by transforming 26 diverse datasets into instruction format and introducing an instruction-aware Query Transformer. The model achieves SOTA zero-shot performance across 13 held-out tasks and demonstrates exceptional capabilities in complex visual reasoning, knowledge-grounded image description, and multi-turn conversations.

## Technical Implementation
- Architecture: Built on BLIP-2 with instruction-aware Query Transformer
- Training: Two-stage approach combining vision-language pre-training and instruction tuning
- Performance: 90.7% accuracy on ScienceQA with image contexts
- Key Feature: Instruction-aware Query Transformer for extracting instruction-tailored features

## Code Example
```python
from lavis.models import load_model_and_preprocess
model, vis_processors, txt_processors = load_model_and_preprocess(
    name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True
)

# Process image and instruction
image = vis_processors["eval"](raw_image).unsqueeze(0)
instruction = "What is unusual about this image?"
response = model.generate({"image": image, "prompt": instruction})
