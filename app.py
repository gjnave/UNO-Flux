# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import json
import os
import subprocess
import platform
from pathlib import Path
from typing import Dict, Optional, Literal

import gradio as gr
import torch

from uno.flux.pipeline import UNOPipeline


def get_examples(examples_dir: str = "assets/examples") -> list:
    examples = Path(examples_dir)
    ans = []
    for example in examples.iterdir():
        if not example.is_dir():
            continue
        with open(example / "config.json") as f:
            example_dict = json.load(f)

        example_list = []

        example_list.append(example_dict["useage"])  # case for
        example_list.append(example_dict["prompt"])  # prompt

        for key in ["image_ref1", "image_ref2", "image_ref3", "image_ref4"]:
            if key in example_dict:
                example_list.append(str(example / example_dict[key]))
            else:
                example_list.append(None)

        example_list.append(example_dict["seed"])

        ans.append(example_list)
    return ans


def get_next_filename(folder="outputs"):
    os.makedirs(folder, exist_ok=True)
    existing_files = [f for f in os.listdir(folder) if f.endswith('.png') and f[:4].isdigit()]
    if not existing_files:
        return os.path.join(folder, "0001.png")
    numbers = [int(f[:4]) for f in existing_files if f[:4].isdigit()]
    next_number = max(numbers) + 1 if numbers else 1
    return os.path.join(folder, f"{next_number:04d}.png")


def open_output_folder():
    folder_path = os.path.abspath("outputs")
    if platform.system() == "Windows":
        os.startfile(folder_path)
    elif platform.system() == "Darwin":
        subprocess.run(["open", folder_path])
    else:
        subprocess.run(["xdg-open", folder_path])
    return "Opened outputs folder"


def create_demo(
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    offload: bool = True,
):
    pipelines: Dict[str, Optional[UNOPipeline]] = {
        "flux-dev": None,
        "flux-dev-fp8": None,
        "flux-schnell": None
    }

    def get_pipeline(model_type: str, offload_setting: bool = offload) -> UNOPipeline:
        if pipelines[model_type] is None:
            pipelines[model_type] = UNOPipeline(model_type, device, offload_setting, only_lora=True, lora_rank=512)
        return pipelines[model_type]

    badges_text = r"""
    <div style="text-align: center; display: flex; justify-content: left; gap: 5px;">
    <a href="https://github.com/bytedance/UNO"><img alt="Build" src="https://img.shields.io/github/stars/bytedance/UNO"></a> 
    <a href="https://bytedance.github.io/UNO/"><img alt="Build" src="https://img.shields.io/badge/Project%20Page-UNO-yellow"></a> 
    <a href="https://arxiv.org/abs/2504.02160"><img alt="Build" src="https://img.shields.io/badge/arXiv%20paper-UNO-b31b1b.svg"></a>
    <a href="https://huggingface.co/bytedance-research/UNO"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Model&color=orange"></a>
    <a href="https://huggingface.co/spaces/bytedance-research/UNO-FLUX"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=demo&color=orange"></a>
    </div>
    """.strip()

    with gr.Blocks() as demo:
        gr.Markdown("Listen to Good Music While You Work: https://music.youtube.com/channel/UCY658vbL6S2zlRomNHoX54Q")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    model_choice = gr.Dropdown(
                        choices=list(pipelines.keys()),
                        value="flux-dev-fp8",
                        label="Model Variant",
                        info="Select a model variant to use"
                    )
                    offload_option = gr.Checkbox(
                        label="Enable Offload",
                        value=offload,
                        info="Offload models to CPU when not in use to save memory"
                    )
                prompt = gr.Textbox(label="Prompt", value="handsome woman in the city")
                with gr.Row():
                    image_prompt1 = gr.Image(label="Ref Img1", visible=True, interactive=True, type="pil")
                    image_prompt2 = gr.Image(label="Ref Img2", visible=True, interactive=True, type="pil")
                    image_prompt3 = gr.Image(label="Ref Img3", visible=True, interactive=True, type="pil")
                    image_prompt4 = gr.Image(label="Ref img4", visible=True, interactive=True, type="pil")

                with gr.Row():
                    with gr.Column():
                        width = gr.Slider(512, 2048, 512, step=20, label="Gneration Width")
                        height = gr.Slider(512, 2048, 512, step=20, label="Gneration Height")
                    with gr.Column():
                        gr.Markdown("📌 Recommended Resolution: 512x512")
                        gr.Markdown(
                            "This model is specifically trained on 512x512 images. "
                            "For best performance and image quality, keep your generation size at 512x512. "
                            "If you need a higher resolution, consider upscaling the output using an external tool."
                        )

                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row():
                        num_steps = gr.Slider(1, 50, 25, step=1, label="Number of steps")
                        guidance = gr.Slider(1.0, 5.0, 4.0, step=0.1, label="Guidance", interactive=True)
                        seed = gr.Number(-1, label="Seed (-1 for random)")

                generate_btn = gr.Button("Generate")
                model_loading_info = gr.Markdown("", visible=False)

            with gr.Column():
                output_image = gr.Image(label="Generated Image")
                download_btn = gr.File(label="Download full-resolution", type="filepath", interactive=False)
                open_folder_btn = gr.Button("Open Outputs Folder")
                folder_status = gr.Markdown("")

            def gradio_generate_wrapper(
                model_type, offload_enabled, prompt, width, height, guidance, num_steps,
                seed, image_prompt1, image_prompt2, image_prompt3, image_prompt4
            ):
                model_loading_info.visible = True
                model_loading_info.value = f"Loading model {model_type}..."
                
                if pipelines[model_type] is not None and pipelines[model_type].offload != offload_enabled:
                    pipelines[model_type] = None
                
                pipeline = get_pipeline(model_type, offload_enabled)
                
                result = pipeline.gradio_generate(
                    prompt, width, height, guidance, num_steps,
                    seed, image_prompt1, image_prompt2, image_prompt3, image_prompt4
                )
                
                output_img, download_path = result
                if output_img is not None:
                    save_path = get_next_filename()
                    output_img.save(save_path)
                    print(f"Image saved to {save_path}")
                
                model_loading_info.visible = False
                return result

            inputs = [
                model_choice, offload_option, prompt, width, height, guidance, num_steps,
                seed, image_prompt1, image_prompt2, image_prompt3, image_prompt4
            ]
            generate_btn.click(
                fn=gradio_generate_wrapper,
                inputs=inputs,
                outputs=[output_image, download_btn],
            )
            
            open_folder_btn.click(
                fn=open_output_folder,
                inputs=[],
                outputs=[folder_status]
            )

        example_text = gr.Text("", visible=False, label="Case For:")
        examples = get_examples("./assets/examples")

        gr.Examples(
            examples=examples,
            inputs=[
                example_text, prompt,
                image_prompt1, image_prompt2, image_prompt3, image_prompt4,
                seed, output_image
            ],
        )

    return demo


if __name__ == "__main__":
    from typing import Literal
    from transformers import HfArgumentParser

    @dataclasses.dataclass
    class AppArgs:
        device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"
        offload: bool = dataclasses.field(
            default=True,
            metadata={"help": "If True, sequantial offload the models(ae, dit, text encoder) to CPU if not used."}
        )
        port: int = 7860
        share: bool = dataclasses.field(
            default=False,
            metadata={"help": "If True, create a publicly shareable link for the app"}
        )

    parser = HfArgumentParser([AppArgs])
    args_tuple = parser.parse_args_into_dataclasses()  # type: tuple[AppArgs]
    args = args_tuple[0]

    demo = create_demo(args.device, args.offload)
    demo.launch(inbrowser=True, share=args.share)
