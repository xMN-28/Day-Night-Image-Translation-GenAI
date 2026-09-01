from __future__ import annotations

import json

import gradio as gr
from PIL import Image

from daynight.inference import translate

HAS_IMAGE_SLIDER = hasattr(gr, "ImageSlider")
GRADIO_MAJOR = int(gr.__version__.split(".", maxsplit=1)[0])


CSS = """
:root { --night: #07111f; --dawn: #ff9c5a; --ice: #dceeff; }
.gradio-container { max-width: 1180px !important; margin: auto !important; }
.hero { padding: 22px 24px; border-radius: 20px; background: linear-gradient(120deg, #07111f 0%, #17233b 55%, #bd5b31 100%); color: white; }
.hero h1 { margin: 0 0 6px 0; font-size: 2.1rem; }
.hero p { opacity: .86; margin: 0; }
.status { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
"""


def run_translation(image: Image.Image, direction: str, model: str, maximum_edge: int):
    try:
        output, metadata = translate(image, direction, model, maximum_edge)
        comparison = (image, output) if HAS_IMAGE_SLIDER else [(image, "Before"), (output, "After")]
        status = "### Translation complete\n```json\n" + json.dumps(metadata, indent=2) + "\n```"
        return output, comparison, status
    except Exception as error:
        raise gr.Error(str(error)) from error


def build_app() -> gr.Blocks:
    block_options = {"title": "LumiCycle — Day ↔ Night"}
    if GRADIO_MAJOR < 6:
        block_options["css"] = CSS
    with gr.Blocks(**block_options) as demo:
        gr.HTML(
            "<section class='hero'><h1>LumiCycle</h1><p>Structure-aware, bidirectional day ↔ night translation trained on unpaired driving scenes.</p></section>"
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Input photograph",
                    type="pil",
                    image_mode="RGB",
                    sources=["upload", "webcam", "clipboard"],
                    height=390,
                )
                with gr.Row():
                    direction = gr.Radio(
                        ["Day → Night", "Night → Day"], value="Day → Night", label="Direction"
                    )
                    model = gr.Dropdown(
                        ["LumiCycle", "CycleGAN", "Turbo reference"],
                        value="LumiCycle",
                        label="Model",
                        info="Turbo is an externally pretrained benchmark.",
                    )
                maximum_edge = gr.Slider(
                    256, 1024, value=768, step=64, label="Maximum inference edge"
                )
                with gr.Row():
                    run = gr.Button("Translate", variant="primary")
                    clear = gr.ClearButton(value="Reset")
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Translated output", type="pil", image_mode="RGB", height=390
                )
                if HAS_IMAGE_SLIDER:
                    slider = gr.ImageSlider(label="Before / after", height=320)
                else:
                    slider = gr.Gallery(label="Before / after", columns=2, rows=1, height=320)
                status = gr.Markdown(
                    "Upload an image and choose a direction.", elem_classes="status"
                )

        run.click(
            run_translation,
            inputs=[input_image, direction, model, maximum_edge],
            outputs=[output_image, slider, status],
            api_name="translate",
            concurrency_limit=1,
        )
        clear.add([input_image, output_image, slider, status])
        gr.Markdown(
            "**Academic note:** LumiCycle and CycleGAN are locally trained project models. The optional Turbo reference is clearly separated and attributed."
        )
    return demo


demo = build_app()


if __name__ == "__main__":
    launch_options = {"server_name": "127.0.0.1", "inbrowser": True}
    if GRADIO_MAJOR >= 6:
        launch_options["css"] = CSS
    demo.queue(default_concurrency_limit=1).launch(**launch_options)
