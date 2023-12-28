import argparse
import gradio as gr

# Custom Modules
import infer

# Argument parser
parser = argparse.ArgumentParser(description="Your description here")
parser.add_argument("--colab", action="store_true", help="Use when running inference from Google Colab")

args = parser.parse_args()

# Create Blocks

# Create interface
inputs = [
    gr.Image(type="filepath", label="Image"),
    gr.Audio(type="filepath", label="Audio"),
    gr.Slider(minimum=1, maximum=60, step=1, value=30, label="FPS"),
    gr.Slider(minimum=0, maximum=160, step=16, value=16, label="Mel Step Size"),
    gr.Slider(minimum=0.0, maximum=2.0, step=0.1, value=1.0, label="GFPGAN Weight")
]
outputs = gr.Video(sources='upload', label="Output")
title = "Lip Wise"

if args.colab:
    gr.Interface(fn=infer.infer_image, inputs=inputs, outputs=outputs, title=title).launch(share=True)
else:
    gr.Interface(fn=infer.infer_image, inputs=inputs, outputs=outputs, title=title).launch()