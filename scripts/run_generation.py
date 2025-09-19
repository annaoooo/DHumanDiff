import os
import argparse
import torch
from PIL import Image

from src.custom_pipelines_worse import StableDiffusionXLCustomPipeline_worse
from src.custom_pipelines_advanced_scale import StableDiffusionXLCustomPipeline
from src.ip_adapter_worse_new import IPAdapterPlusXL_worse
from src.ip_adapter_new_scale import IPAdapterPlusXL_scale


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    # Fixed paths (you can change if needed)
    SD = "model/stable-diffusion-xl-base-1.0"
    image_encoder_path = "model/ipadapter_model/image_encoder"
    clip_encoder = "model/clip-vit-large-patch14"
    pre_trained = "model/DHumanDiff"

    ip_adapter_ca = os.path.join(pre_trained, f"ip_adapter_ca.pt")
    img_proj1 = os.path.join(pre_trained, f"face_projection.pt")
    img_proj2 = os.path.join(pre_trained, f"face_projection2.pt")
    mapping_net = os.path.join(pre_trained, f"mapping.pt")

    # Load pipelines
    pipe = StableDiffusionXLCustomPipeline.from_pretrained(SD, torch_dtype=dtype, add_watermarker=False)
    pipe_worse = StableDiffusionXLCustomPipeline_worse.from_pretrained(SD, torch_dtype=dtype, add_watermarker=False)

    # Load adapters
    ip_model_worse = IPAdapterPlusXL_worse(
        pipe_worse, image_encoder_path, clip_encoder,
        mapping_net, ip_adapter_ca, img_proj1, img_proj2,
        device, num_tokens=16, scale_factor=0.7
    )
    ip_model_worse.load_ip_adapter()

    ip_model = IPAdapterPlusXL_scale(
        pipe, image_encoder_path, clip_encoder,
        mapping_net, ip_adapter_ca, img_proj1, img_proj2,
        device, num_tokens=16, scale_factor=1.0
    )
    ip_model.load_ip_adapter()
    unet_worse = pipe_worse.unet

    # Load input images
    image1 = Image.open(args.image1).convert("RGB")
    image2 = Image.open(args.image2).convert("RGB")

    # Output dirs
    os.makedirs("output", exist_ok=True)
    mask_dir = os.path.join("output", "masks")
    gen_dir = os.path.join("output", "generated")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)

    # First stage
    images = ip_model_worse.generate(
        person_idx1=args.person_idx1,
        person_idx2=args.person_idx2,
        pil_image1=image1,
        pil_image2=image2,
        num_samples=1,
        num_inference_steps=50,
        seed=42,
        prompt=args.prompt,
        height=512,
        width=512,
        folder_name=mask_dir,
        top=90,
        scale=0.7,
    )

    # Save intermediate
    inter_file = os.path.join(mask_dir, "intermediate.png")
    images[0].save(inter_file)
    print(f"Intermediate saved -> {inter_file}")

    # Second stage
    final_images = ip_model.generate_advanced(
        person_idx1=args.person_idx1,
        person_idx2=args.person_idx2,
        pil_image1=image1,
        pil_image2=image2,
        num_samples=1,
        num_inference_steps=50,
        seed=42,
        prompt=args.prompt,
        height=512,
        width=512,
        prior_mask=mask_dir,
        unet_worse=unet_worse,
        use_worse_model=False,
        scale=1.0,
    )

    final_file = os.path.join(gen_dir, "final.png")
    final_images[0].save(final_file)
    print(f"Final image saved -> {final_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image1", type=str, required=True, help="Path to person A image")
    parser.add_argument("--image2", type=str, required=True, help="Path to person B image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--person_idx1", type=int, required=True, help="Index of person A")
    parser.add_argument("--person_idx2", type=int, required=True, help="Index of person B")

    args = parser.parse_args()
    main(args)
