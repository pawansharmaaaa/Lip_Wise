# This file is a part of https://github.com/pawansharmaaaa/Lip_Wise/ repository.
import gradio as gr

import helpers.file_check as fc
from modules.face_resolver import FaceResolver

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

# Create interface

with gr.Blocks(title='Lip-Wise', theme=theme, css = fc.CSS_FILE_PATH) as demo:
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
    with gr.Tab(label="Face-Selector", elem_id="tab", elem_classes=["tabs"]):
        with gr.Row(elem_classes=["row"]):
            with gr.Column():
                with gr.Accordion(label="Input Video", open=True, elem_classes=["inp_group", "accordion"]):
                    video_input = gr.File(file_count='single', 
                                        label="Video",
                                        container=True)
                    video_preview = gr.Video(label="Video")

                with gr.Column():
                    frame = gr.Image(label="Frame",
                                     scale=1)
                    frame_index = gr.Slider(minimum=0, 
                                            maximum=100, 
                                            step=1, 
                                            label="Frame Index",
                                            container=True,
                                            interactive=True)
                    
            with gr.Column(scale=1):
                with gr.Column(scale=1):
                    metadata = gr.HighlightedText(label="Metadata")
                        
                    faces = gr.Gallery(label="Select Reference Face",
                                       scale=1,
                                       elem_classes=["input"],
                                       allow_preview=False,
                                       object_fit="contain")
                    
                    similarity_strength = gr.Slider(value=1.2,
                                                    minimum=0.1,
                                                    maximum=2,
                                                    step=0.1,
                                                    label="Similarity Strength",
                                                    container=True)
                    
                    process = gr.Button(value="Generate Preview",
                                        variant="primary", 
                                        elem_id="gen-button",
                                        size='sm')
            
        with gr.Row():
            with gr.Column():
                preview = gr.Video(sources='upload', 
                                   label="Preview", 
                                   elem_classes=["output"],
                                   container=True)
                save = gr.Button(value="Save Data", 
                                 interactive=False, 
                                 variant="secondary", 
                                 elem_id="clear-button")
                
        ### Function and Event Mapping
        fr = FaceResolver()
        
        def update_selection(evt: gr.SelectData):
            fr.save_embedding(evt.index)

        def populate_metadata():
            metadata = [
                ("FPS", str(fr.fps)),
                ("RESOLUTION", f"{str(fr.width)}x{str(fr.height)}"),
                ("TOTAL FRAMES", str(fr.frame_count))
            ]
            return metadata
        
        def return_path():
            return fr.video_path
        
        def update_slider_max(video_input):
            fr.process_video(video_input)
            return gr.update(maximum=(fr.frame_count-1))
        
        def show_save_button():
            return gr.update(interactive=True)
                
        video_input.change(fn=update_slider_max, 
                           inputs=[video_input],
                           outputs=[frame_index]
                           ).then(return_path, 
                                  outputs=[video_preview]
                           ).then(populate_metadata, 
                                  outputs=[metadata]
                           ).then(
                                fr.update_frame, 
                                inputs=[frame_index], 
                                outputs=[frame, faces]
                        )
        
        frame_index.change(fn=fr.update_frame, 
                           inputs=[frame_index], 
                           outputs=[frame, faces])
        
        faces.select(fn=update_selection)
        
        process.click(fn=fr.get_preview,
                    inputs=[similarity_strength],
                    outputs=[preview]).then(
                        show_save_button,
                        outputs=[save]
                    )
        
        save.click(fn=fr.save_state)
          
demo.launch()
