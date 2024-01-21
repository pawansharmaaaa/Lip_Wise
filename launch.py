# This file is a part of https://github.com/pawansharmaaaa/Lip_Wise/ repository.

import argparse
import gradio as gr

# Custom Modules
import infer

# Argument parser
parser = argparse.ArgumentParser(description="Your description here")
parser.add_argument("--colab", action="store_true", help="Use when running inference from Google Colab")

args = parser.parse_args()

# Create interface
inputs_for_image = [
    gr.Image(type="filepath", label="Image"),
    gr.Audio(type="filepath", label="Audio"),
    gr.Number(value=0, label="Padding: Increase if getting black outlines"),
    gr.Checkbox(label = "Perform 3D_alignment?"),
    gr.Radio(["GFPGAN", "CodeFormer"], value='CodeFormer', label="Face Restorer"),
    gr.Slider(minimum=1, maximum=60, step=1, value=30, label="FPS"),
    gr.Number(value=16, label="Mel Step Size", interactive=False),
    gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.3, label="Weight")
]
output_for_image = gr.Video(sources='upload', label="Output")

image_Interface = gr.Interface(fn=infer.infer_image, inputs=inputs_for_image, outputs=output_for_image)

inputs_for_video = [
    gr.Video(sources='upload',label="Video"),
    gr.Audio(type="filepath", label="Audio"),
    gr.Number(value=0, label="Padding: Increase if getting black outlines"),
    gr.Radio(["GFPGAN", "CodeFormer"], value='CodeFormer', label="Face Restorer"),
    gr.Number(value=16, label="Mel Step Size", interactive=False),
    gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.3, label="Weight")
]
output_for_video = gr.Video(sources='upload', label="Output")

video_Interface = gr.Interface(fn=infer.infer_video, inputs=inputs_for_video, outputs=output_for_video)

# Run interface
ui = gr.TabbedInterface([image_Interface, video_Interface], ['Process Image', 'Process Video'],title="Lip-Wise")

if args.colab:
    ui.queue().launch(share=True)
else:
    ui.queue().launch()