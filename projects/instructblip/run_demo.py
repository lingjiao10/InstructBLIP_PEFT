import gradio as gr
import torch
import argparse
from omegaconf import OmegaConf
from lavis.common.config import Config
from lavis.common.registry import registry
from lavis.models import load_model_and_preprocess
from lavis.models import load_preprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model-name", default="blip2_vicuna_instruct")
    parser.add_argument("--model-type", default="vicuna7b")
    parser.add_argument("--model-ftckp", default="")
    args = parser.parse_args()

    image_input = gr.Image(type="pil")

    min_len = gr.Slider(
        minimum=1,
        maximum=50,
        value=1,
        step=1,
        interactive=True,
        label="Min Length",
    )

    max_len = gr.Slider(
        minimum=10,
        maximum=500,
        value=250,
        step=5,
        interactive=True,
        label="Max Length",
    )

    sampling = gr.Radio(
        choices=["Beam search", "Nucleus sampling"],
        value="Beam search",
        label="Text Decoding Method",
        interactive=True,
    )

    top_p = gr.Slider(
        minimum=0.5,
        maximum=1.0,
        value=0.9,
        step=0.1,
        interactive=True,
        label="Top p",
    )

    beam_size = gr.Slider(
        minimum=1,
        maximum=10,
        value=5,
        step=1,
        interactive=True,
        label="Beam Size",
    )

    len_penalty = gr.Slider(
        minimum=-1,
        maximum=2,
        value=1,
        step=0.2,
        interactive=True,
        label="Length Penalty",
    )

    repetition_penalty = gr.Slider(
        minimum=-1,
        maximum=3,
        value=1,
        step=0.2,
        interactive=True,
        label="Repetition Penalty",
    )


    prompt_textbox = gr.Textbox(label="Prompt:", placeholder="prompt", lines=2)

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    print('Loading model...')

    if(args.model_ftckp==""):
        model, vis_processors, _ = load_model_and_preprocess(
            name=args.model_name,
            model_type=args.model_type,
            is_eval=True,
            device=device,
        )
    else:
    # if args.model_ftckp is not "":
        #read cfg 
        # cfg = Config(args.cfg)


        checkpoint_path = args.model_ftckp
        # model_config = cfg.model_cfg

        

        print("Loading checkpoint from {}.".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        cfg = OmegaConf.create(checkpoint["config"])
        model_config = Config.build_model_config(cfg).model
        print("--------------model_config")
        print(OmegaConf.to_yaml(model_config))
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config)

        try:
            model.load_state_dict(checkpoint["model"])
        except RuntimeError as e:
            print(
                """
                Key mismatch when loading checkpoint. This is expected if only part of the model is saved.
                Trying to load the model with strict=False.
                """
            )
            model.load_state_dict(checkpoint["model"], strict=False)

        preprocess_cfg = cfg.preprocess
        vis_processors, txt_processors = load_preprocess(preprocess_cfg)
        model.to(device)

    print('Loading model done!')

    def inference(image, prompt, min_len, max_len, beam_size, len_penalty, repetition_penalty, top_p, decoding_method, modeltype):
        use_nucleus_sampling = decoding_method == "Nucleus sampling"
        print(image, prompt, min_len, max_len, beam_size, len_penalty, repetition_penalty, top_p, use_nucleus_sampling)
        image = vis_processors["eval"](image).unsqueeze(0).to(device)

        samples = {
            "image": image,
            "prompt": prompt,
        }

        output = model.generate(
            samples,
            length_penalty=float(len_penalty),
            repetition_penalty=float(repetition_penalty),
            num_beams=beam_size,
            max_length=max_len,
            min_length=min_len,
            top_p=top_p,
            use_nucleus_sampling=use_nucleus_sampling,
        )

        print("model output: ", output)
        if(output[0]!=''):
            return output[0]
        else:
            return "Sorry, I don't know the answer."

    gr.Interface(
        fn=inference,
        inputs=[image_input, prompt_textbox, min_len, max_len, beam_size, len_penalty, repetition_penalty, top_p, sampling],
        outputs="text",
        allow_flagging="never",
    ).launch(server_name="192.168.30.153")
