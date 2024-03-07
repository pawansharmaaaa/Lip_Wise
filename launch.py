# This file is a part of https://github.com/pawansharmaaaa/Lip_Wise/ repository.

import argparse
import gradio as gr

# Custom Modules
import infer
from helpers import file_check

# Argument parser
parser = argparse.ArgumentParser(description="Your description here")
parser.add_argument("--colab", action="store_true", help="Use when running inference from Google Colab")

args = parser.parse_args()

bg_upscalers = list(file_check.REAL_ESRGAN_MODEL_URL.keys())

# Theme
theme = gr.themes.Base(
    primary_hue=gr.themes.Color(c100="#efe7ff", 
                                c200="#decefe", 
                                c300="#ccb7fd", 
                                c400="#ba9ffc", 
                                c50="#ffffff", 
                                c500="#a688fa", 
                                c600="#836bc3", 
                                c700="#61508e", 
                                c800="#41365d", 
                                c900="#241e30", 
                                c950="#25242a"),
    secondary_hue=gr.themes.Color(c100="#e2e1e4", 
                                  c200="#c6c5c9", 
                                  c300="#aba9b0", 
                                  c400="#908d96", 
                                  c50="#ffffff", 
                                  c500="#76737e", 
                                  c600="#5e5b64", 
                                  c700="#46444b", 
                                  c800="#302f33", 
                                  c900="#1b1b1d", 
                                  c950="#25242a"),
    neutral_hue=gr.themes.Color(c100="#e1e1e1", 
                                c200="#c4c4c4", 
                                c300="#a7a7a7", 
                                c400="#8c8c8c", 
                                c50="#ffffff", 
                                c500="#717171", 
                                c600="#5a5a5a", 
                                c700="#434343", 
                                c800="#2e2e2e", 
                                c900="#25242a", 
                                c950="#1b1b1b"),
    spacing_size="md",
    radius_size="lg",
).set(
    shadow_drop='*shadow_inset',
    shadow_drop_lg='*button_shadow_hover',
    block_info_text_size='md',
    block_info_text_weight='800',
    block_info_text_color_dark="#C494ACFF",
    block_title_text_color_dark="#FFFFFFFF",
    slider_color_dark="#B12805FF",
    panel_border_color_dark="#B12805FF",
    loader_color_dark="#C494ACFF",
    # body_background_fill="radial-gradient( circle farthest-corner at -4% -12.9%,  rgba(255,255,255,1) 0.3%, rgba(255,255,255,1) 90.2% );",
    # body_background_fill_dark= "linear-gradient(315deg, #0cbaba 0%, #380036 74%);"
    )

head_html = f'''
    <head class>
    <meta author="Pawan Sharma">
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
'''

def render_dd(bg_upscale):
    
    return gr.Dropdown(
                        choices=bg_upscalers,
                        label="REALESRGAN Model",
                        value="RealESRGAN_x2plus",
                        info="Choose the model to use for upscaling the background.",
                        visible=bg_upscale
                    )

def render_weight(face_restorer):
    if face_restorer == "CodeFormer":
        return gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=0.6,
                            label="CodeFormer Weight",
                            info="0 for better quality, 1 for better identity.",
                            visible=True,
                            scale=2,
                            elem_classes=["option"]
                        )
    elif face_restorer == "GFPGAN":
        return gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=0.5,
                            label="GFPGAN Weight",
                            info="0 for better identity, 1 for better quality.",
                            visible=True,
                            scale=2,
                            elem_classes=["option"]
                        )
    elif face_restorer == "RestoreFormer":
        return gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=0.5,
                            label="RestoreFormer Weight",
                            info="0 for better identity, 1 for better quality.",
                            visible=True,
                            scale=2,
                            elem_classes=["option"]
                        )
    elif face_restorer == "None":
        return gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=0.5,
                            label="None Weight",
                            info="0 for better identity, 1 for better quality.",
                            visible=False
                        )

# Create interface

with gr.Blocks(title='Lip-Wise', theme=theme, css = file_check.CSS_FILE_PATH) as ui:
    with gr.Row(elem_classes=["row"]):
        gr.HTML(
            '''
            <header>
            <div class="header-left">
                <h1>Lip Wise</h1>
                <h2>Wise Enhancements for Wav2Lip</h2>
            </div>
            <div class="header-right">
                <img src="https://github.com/pawansharmaaaa/Lip_Wise/assets/56242483/5bc1b8af-879a-414b-b54a-db605a53c8f7" alt="Logo">
            </div>
            </header>
            '''
        )

    with gr.Tab(label="Process Image", elem_id="tab", elem_classes=["tabs"]):
        with gr.Row(elem_classes=["row"]):
            with gr.Column():
                # gr.Markdown("# INPUTS")
                with gr.Accordion(label="Input Image and Audio", open=True, elem_classes=["inp_group", "accordion"]):
                    image_input = gr.Image(type="filepath", 
                                           label="Image", 
                                           container=True)
                    
                    audio_input = gr.Audio(type="filepath", 
                                           label="Audio", 
                                           container=True)
                    
            with gr.Column():
                # gr.Markdown("# OUTPUT")
                with gr.Column(scale=2):
                    image_output = gr.Video(sources='upload', 
                                            label="Output", 
                                            elem_classes=["output"],
                                            container=True)
                with gr.Column(scale=3):
                    gr.HTML(
                            '''
                            <div class="separator"></div>
                            ''')
                with gr.Column(variant="panel", scale=2):
                    process = gr.Button(value="Process Image",
                                        variant="primary", 
                                        elem_id="gen-button")
                    clear = gr.ClearButton([image_input, audio_input, image_output],
                                            value="Clear",
                                            variant="secondary",
                                            elem_id="clear-button")

        with gr.Accordion(label="OPTIONS", open=True, elem_classes=["opt_group", "accordion"]):
            with gr.Group():
                with gr.Column(variant="panel"):
                    with gr.Row():
                        gan = gr.Checkbox(label = "Use Wav2Lip_GAN?", 
                                            value=False, 
                                            info="This will use Wav2Lip_GAN instead of Wav2Lip. May get better results in some cases", 
                                            interactive=True,
                                            elem_classes=["option"])
                        
                        alignment = gr.Checkbox(label = "Perform 3D_alignment", 
                                                info = "This will improve the quality of the lip sync, but the output will be different from the original video.",
                                                elem_classes=["option"])
                        
                    with gr.Row():
                        fps = gr.Slider(minimum=1, 
                                        maximum=60, 
                                        step=1, 
                                        value=25, 
                                        label="FPS", 
                                        info="Desired Frames per second (FPS) of the output video.",
                                        elem_classes=["option"])
                        
                        padding = gr.Slider(minimum=0, 
                                            maximum=60, 
                                            step=1, 
                                            value=0, 
                                            label="Padding", 
                                            info="Increase if getting black outlines. The Value is in Pixels.",
                                            elem_classes=["option"])
                    
                with gr.Column(variant="panel"):
                    with gr.Row():
                        face_restorer = gr.Radio(["GFPGAN", "CodeFormer", "RestoreFormer", "VQFR1","VQFR2","None"], 
                                                    value='CodeFormer', 
                                                    label="Face Restorer", 
                                                    info="GFPGAN is faster, but CodeFormer is more accurate.", # Needs Change
                                                    interactive=True,
                                                    elem_classes=["option"])
                        
                        weight = gr.Slider(minimum=0.0, 
                                            maximum=1.0, 
                                            step=0.1, 
                                            value=0.6, 
                                            label="CodeFormer Weight", 
                                            info="0 for better quality, 1 for better identity.",
                                            scale=2,
                                            elem_classes=["option"])
                    
                    with gr.Row():
                            upscale_bg = gr.Checkbox(label = "Upscale Background with REALESRGAN",
                                                    value=False, 
                                                    info="This will improve the quality of the video, but will take longer to process.",
                                                    elem_classes=["option"])
                            
                            bg_model = gr.Dropdown(choices=bg_upscalers, 
                                                    label="REALESRGAN Model", 
                                                    value="RealESRGAN_x2plus", 
                                                    info="Choose the model to use for upscaling the background.", 
                                                    visible=False,
                                                    scale=2,
                                                    elem_classes=["option"])

                            mel_step_size = gr.Number(value=16, 
                                                        label="Mel Step Size", 
                                                        interactive=False, 
                                                        visible=False,
                                                        elem_classes=["option"])
                        
                # Event Triggers
                upscale_bg.select(render_dd, 
                                    upscale_bg, 
                                    bg_model)
                face_restorer.select(render_weight, 
                                        face_restorer, 
                                        weight)

            process.click(infer.infer_image, 
                            [image_input, audio_input, padding, alignment, face_restorer, fps, mel_step_size, weight, upscale_bg, bg_model, gan],
                            [image_output],
                            show_progress="full",
                            trigger_mode="once")
            
            

    with gr.Tab(label="Process Video", elem_id="tab", elem_classes=["tabs"]):
        with gr.Row(elem_classes=["row"]):
            with gr.Column():
                # gr.Markdown("# INPUTS")
                with gr.Accordion("Input Video and Audio", open=True, elem_classes=["inp_group", "accordion"]):
                    video_input = gr.Video(sources='upload',
                                           label="Video")
                    audio_input = gr.Audio(type="filepath", 
                                           label="Audio")
                    
            with gr.Column():
                # gr.Markdown("# OUTPUT")
                with gr.Column(scale=2):
                    video_output = gr.Video(sources='upload', 
                                            label="Output", 
                                            elem_classes=["output"],
                                            container=True)
                
                with gr.Column(scale=3):
                    gr.HTML(
                            '''
                            <div class="separator"></div>
                            ''')
                
                with gr.Column(variant="panel"):
                    process = gr.Button(value="Process Video", 
                                        variant="primary", 
                                        elem_id="gen-button")
                    clear = gr.ClearButton([video_input, audio_input, video_output],
                                            value="Clear",
                                            variant="secondary",
                                            elem_id="clear-button")
                
        with gr.Accordion(label="OPTIONS", open=True, elem_classes=["opt_group", "accordion"]):
            with gr.Group():
                with gr.Column():
                    with gr.Row():
                        gan = gr.Checkbox(label = "Use Wav2Lip_GAN", 
                                        value=False, 
                                        info="This will use Wav2Lip_GAN instead of Wav2Lip. May get better results in some cases", 
                                        interactive=True)
                        loop = gr.Checkbox(label = "Loop Video", 
                                        value=False, 
                                        info="This will loop the video to the length of the audio file.", 
                                        interactive=True)
                           
                    padding = gr.Slider(minimum=0, 
                                        maximum=60, 
                                        step=1, 
                                        value=0, 
                                        label="Padding", 
                                        info="Increase if getting black outlines. The Value is in Pixels.")

                with gr.Column():
                    with gr.Row():
                        face_restorer = gr.Radio(["GFPGAN", "CodeFormer", "RestoreFormer", "VQFR1","VQFR2","None"], 
                                                    value='CodeFormer', 
                                                    label="Face Restorer", 
                                                    info="GFPGAN is faster, but CodeFormer is more accurate.")
                        
                        mel_step_size = gr.Number(value=16, 
                                                    label="Mel Step Size", 
                                                    interactive=False, 
                                                    visible=False)
                        
                        weight = gr.Slider(minimum=0.0, 
                                            maximum=1.0, 
                                            step=0.1, 
                                            value=0.6, 
                                            label="CodeFormer Weight", 
                                            info="0 for better quality, 1 for better identity.",
                                            scale=2)
                
                    with gr.Row():
                        upscale_bg = gr.Checkbox(label = "Upscale Background with REALESRGAN", 
                                                    value=False, 
                                                    info="This will improve the quality of the video, but will take longer to process.")
                        
                        bg_model = gr.Dropdown(choices=bg_upscalers, 
                                                label="REALESRGAN Model", 
                                                value="RealESRGAN_x2plus", 
                                                info="Choose the model to use for upscaling the background.", 
                                                visible=False,
                                                scale=2)
                    
                    # Event Triggers
                    upscale_bg.select(render_dd, upscale_bg, bg_model)
                    face_restorer.select(render_weight, face_restorer, weight)

                process.click(infer.infer_video, 
                              [video_input, audio_input, padding, face_restorer, mel_step_size, weight, upscale_bg, bg_model, gan, loop],
                              [video_output],
                              trigger_mode="once",
                              show_progress="full")

    with gr.Tab(label="Guide", elem_id="tab", elem_classes=["tabs"]):
        with gr.Accordion(label="Tips For Better Results", open=True, elem_classes=["guide"]):
            gr.Markdown(
            """
            > - Optimal performance is achieved with a **clear image** featuring a person facing the camera, regardless of head angle. However, avoid **tilting in the z-direction** (3D-alignment can address this with certain considerations).
            > - Ensure the image contains only **one person** with a prominently visible face.
            > - Clear audio devoid of background noise enhances results significantly.
            > - Note that **higher image resolution** necessitates **additional processing time**.
            """,
            line_breaks=True)

        with gr.Accordion(label="Model Selection", open=True, elem_classes=["guide"]):
            gr.Markdown(
            """
            > ##### **CODEFORMER:**
            >**Recommended Weight:** `0 for better quality, 1 for better identity.`
            >>- CodeFormer employs a transformative architecture to restore facial features.
            >>- While relatively slower, it boasts **higher accuracy**.
            >>- Generally delivers superior results while **preserving skin texture**.
            >>- In cases of peculiar artifacts, especially around the nose, consider using GFPGAN.
            
            > ##### **GFPGAN:**
            >**Recommended Weight:** `0 for better identity, 1 for better quality.`
            >>- GFPGAN, a faster model, relies on a GAN-based framework for facial restoration.
            >>- Suggested for use **when CodeFormer exhibits undesirable artifacts**.
            >>- However, it often sacrifices skin texture fidelity.
            """,
            line_breaks=True)

        with gr.Accordion(label="3D Alignment", open=True, elem_classes=["guide"]):
            gr.Markdown(
            """
            > Enabling this feature **transforms** the image to ensure the person faces the camera directly. While enhancing lip sync quality, the output may diverge from the original video.
            """,
            line_breaks=True)

        with gr.Accordion(label="Background Upscaling", open=True, elem_classes=["guide"]):
            gr.Markdown(
            """
            > - Activating this feature **enhances video quality** but prolongs processing time.
            > - For most scenarios, RealESRGAN_x2plus is preferable due to its comparative speed.
            > - Optimal results are achieved when combined with CodeFormer, except in cases of nose-related artifacts.
            > - This feature effectively **eliminates video flickering**.
            """,
            line_breaks=True)

if args.colab:
    ui.queue().launch(share=True)
else:
    ui.queue().launch()