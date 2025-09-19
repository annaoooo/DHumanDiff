import os
from typing import List
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTokenizer

from functions import ProjPlusModel
from insightface.app import FaceAnalysis

from utils import is_torch2_available, get_generator
from torchvision import transforms
if is_torch2_available():
    from attention_processor_faceid import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from attention_processor_faceid import (
        CNAttnProcessor2_0 as CNAttnProcessor,
    )
    from attention_processor_faceid import (
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
    
    from attention_processor_faceid import IPAttnProcessor2_0_dual_ca,IPAttnProcessor2_0_dual_ca_worse

else:
    from attention_processor_faceid import AttnProcessor, CNAttnProcessor, IPAttnProcessor
from resampler import Resampler
from mapping import MLP
from model_clip import CLIPTextModel



class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )
        
    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens



class IPAdapter:
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()
        
        
        
        self.load_ip_adapter()

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def set_ip_adapter(self,scale_factor):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = IPAttnProcessor2_0_dual_ca_worse(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=scale_factor,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def generate(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        
        

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)
        
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


class IPAdapterPlusXL_worse(IPAdapter):
    """SDXL"""
    def __init__(self, sd_pipe, image_encoder_path, subject_img_encoder, mapping_path, ip_ca_ckpt, img_proj_ckpt, img_proj2_ckpt, device, num_tokens=4, scale_factor=0.5):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.subject_img_encoder_path = subject_img_encoder
        self.mapping_path = mapping_path
        self.ip_ca_ckpt = ip_ca_ckpt
        self.img_proj_ckpt = img_proj_ckpt
        self.img_proj_ckpt2 = img_proj2_ckpt
        self.num_tokens = num_tokens

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter(scale_factor)

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()
        
        self.subject_img_encoder = CLIPVisionModelWithProjection.from_pretrained(self.subject_img_encoder_path).to(self.device, dtype=torch.float16)
        
        self.mapping_model = MLP(in_dim=1792, out_dim=768, hidden_dim=1024, use_residual=False).to(self.device, dtype=torch.float16)
        
        self.text_encoder_custom = CLIPTextModel.from_pretrained(self.subject_img_encoder_path).to(self.device, dtype=torch.float16)
        
        self.tokenizer = CLIPTokenizer.from_pretrained("model/stable-diffusion-xl-base-1.0", subfolder="tokenizer")
        
        # �ڶ�������ӳ��ģ��
        self.image_proj_model2 = ProjPlusModel(
            cross_attention_dim=1024,  
            id_embeddings_dim=512,
            clip_embeddings_dim=1280,
            num_tokens=1,
        ).to(self.device, dtype=torch.float16)
        # FaceID
        self.app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        
    def init_proj(self):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def load_ip_adapter(self):
        
        ip_ca_state_dict = torch.load(self.ip_ca_ckpt, map_location="cpu")
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(ip_ca_state_dict, strict=True)
        
        img_proj_state_dict = torch.load(self.img_proj_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(img_proj_state_dict, strict=True)
        
        img_proj_state_dict2 = torch.load(self.img_proj_ckpt2, map_location="cpu")
        self.image_proj_model2.load_state_dict(img_proj_state_dict2, strict=True)
        
        mapping_state_dict = torch.load(self.mapping_path, map_location="cpu")
        self.mapping_model.load_state_dict(mapping_state_dict, strict=True)
        
        print("IP Adapter and Image Proj and Mapping Model loaded successfully!!!")

    @torch.inference_mode()
    def get_prepare_faceid(self, face_image):
        faceid_image = np.array(face_image)
        faces = self.app.get(faceid_image)
        if faces==[]:
            faceid_embeds = torch.zeros_like(torch.empty((1, 512)))
        else:
            faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)

        return faceid_embeds

    @torch.inference_mode()
    def get_image_embeds(self, pil_image1, pil_image2):
        if isinstance(pil_image1, Image.Image):
            pil_image1 = [pil_image1]
        if isinstance(pil_image2, Image.Image):
            pil_image2 = [pil_image2]
        
        clip_image1 = self.clip_image_processor(images=pil_image1, return_tensors="pt").pixel_values  # [1,3,224,224]
        clip_image2 = self.clip_image_processor(images=pil_image2, return_tensors="pt").pixel_values
        
        clip_image1 = clip_image1.to(self.device, dtype=torch.float16)
        clip_image_embeds1 = self.image_encoder(clip_image1, output_hidden_states=True).hidden_states[-2] # torch.Size([1, 257, 1280])
        clip_image2 = clip_image2.to(self.device, dtype=torch.float16)
        clip_image_embeds2 = self.image_encoder(clip_image2, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds1 = self.image_proj_model(clip_image_embeds1) # torch.Size([1, 16, 2048])
        image_prompt_embeds2 = self.image_proj_model(clip_image_embeds2)
        
        image_prompt_embeds = torch.cat([image_prompt_embeds1, image_prompt_embeds2], dim=1) # torch.Size([1, 32, 2048])
        
        uncond_clip_image_embeds1 = self.image_encoder(
            torch.zeros_like(clip_image1), output_hidden_states=True
        ).hidden_states[-2]
        uncond_clip_image_embeds2 = self.image_encoder(
            torch.zeros_like(clip_image2), output_hidden_states=True
        ).hidden_states[-2]
        
        uncond_image_prompt_embeds1 = self.image_proj_model(uncond_clip_image_embeds1)
        uncond_image_prompt_embeds2 = self.image_proj_model(uncond_clip_image_embeds2)
        
        uncond_image_prompt_embeds = torch.cat([uncond_image_prompt_embeds1, uncond_image_prompt_embeds2], dim=1)
        return image_prompt_embeds, uncond_image_prompt_embeds
    
    
    def get_sub_image_embeds(self, pil_image1, pil_image2):
        if isinstance(pil_image1, Image.Image):
            pil_image1 = [pil_image1]
        if isinstance(pil_image2, Image.Image):
            pil_image2 = [pil_image2]
        
        clip_image1 = self.clip_image_processor(images=pil_image1, return_tensors="pt").pixel_values  # torch.Size([1, 3, 224, 224])
        clip_image2 = self.clip_image_processor(images=pil_image2, return_tensors="pt").pixel_values
        
        clip_image1 = clip_image1.to(self.device, dtype=torch.float16)
        clip_image_embeds1 = self.subject_img_encoder(clip_image1, output_hidden_states=True).last_hidden_state[:, 0, :]  # torch.Size([1, 1024])
        clip_image2 = clip_image2.to(self.device, dtype=torch.float16)
        clip_image_embeds2 = self.subject_img_encoder(clip_image2, output_hidden_states=True).last_hidden_state[:, 0, :]
    
        
        return clip_image_embeds1, clip_image_embeds2
    
    def get_faceid_embeds(self, pil_image1, pil_image2):
        if isinstance(pil_image1, Image.Image):
            pil_image1 = [pil_image1]
        if isinstance(pil_image2, Image.Image):
            pil_image2 = [pil_image2]
        
        faceid_embeds_1 = self.get_prepare_faceid(face_image=pil_image1[0])
        faceid_embeds_2 = self.get_prepare_faceid(face_image=pil_image2[0])



        
        return faceid_embeds_1, faceid_embeds_2
    
    @torch.inference_mode()   
    def get_image_embeds2(self, faceid_embeds, face_image, s_scale, shortcut=False):
        # ��� clip image embedding 
        clip_image = self.clip_image_processor(images=face_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]


        faceid_embeds = faceid_embeds.to(self.device, dtype=torch.float16)
        image_prompt_tokens = self.image_proj_model2(faceid_embeds, clip_image_embeds, shortcut=shortcut, scale=s_scale)


        return image_prompt_tokens
    

    def generate(
        self,
        person_idx1,
        person_idx2,
        pil_image1,
        pil_image2,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        height=512,
        width=512,
        folder_name=None,
        top=None,
        **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image1, Image.Image) else len(pil_image1)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image1,pil_image2)  # torch.Size([1, 32, 2048])
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        
        # ʹ���µ�embedding �滻 cls embedding
        # person1_subject_embeds, person2_subject_embeds = self.get_sub_image_embeds(pil_image1, pil_image2)  # torch.Size([1, 1024])
        
        text_input_ids = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids  # torch.Size([1, 77])
        
        pre_text_token_embeds = self.text_encoder_custom.get_input_embeddings()(text_input_ids.to(self.device))  # torch.Size([1, 77, 768])
         
        # ��ȡfaceid_feature
        person1_faceid_embeds , person2_faceid_embeds = self.get_faceid_embeds(pil_image1, pil_image2)
        
        # ��faceid��clip�����ں�
        # 5. Prepare the input ID images
        prompt_tokens_faceid1 = self.get_image_embeds2(person1_faceid_embeds, face_image=pil_image1, s_scale=1.0, shortcut=False)
        prompt_tokens_faceid2 = self.get_image_embeds2(person2_faceid_embeds, face_image=pil_image2, s_scale=1.0, shortcut=False)
        prompt_tokens_faceid1 = prompt_tokens_faceid1.squeeze(1)  # torch.Size([1, 1, 1024]) --> torch.Size([1, 1024])
        prompt_tokens_faceid2 = prompt_tokens_faceid2.squeeze(1)  # torch.Size([1, 1024])

        person1_hybrid = torch.cat([pre_text_token_embeds[:,person_idx1,:], prompt_tokens_faceid1], dim=1)  # torch.Size([1, 1792]) 768+1024
        person2_hybrid = torch.cat([pre_text_token_embeds[:,person_idx2,:], prompt_tokens_faceid2], dim=1)
        persons_hybrid = torch.cat([person1_hybrid, person2_hybrid], dim=0)
        person_embeds = self.mapping_model(persons_hybrid)  # torch.Size([2, 768])
        
        person1_embeds = person_embeds[0, :] # torch.Size([768])
        person2_embeds = person_embeds[1, :]
        
        pre_text_token_embeds[:,person_idx1,:] = person1_embeds  # torch.Size([1, 77, 768])
        pre_text_token_embeds[:,person_idx2,:] = person2_embeds
        
        prompt_embeds_first = self.text_encoder_custom(inputs_embeds=pre_text_token_embeds)[0]
        
        
        with torch.inference_mode():
            (
                prompt_embeds,  # # torch.Size([1, 77, 2048])
                negative_prompt_embeds,
                pooled_prompt_embeds,  # torch.Size([1, 1280])
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            
            # using hybrid embedding to replace the embedding of first text encoder
            prompt_embeds[:,:,:prompt_embeds_first.shape[-1]] = prompt_embeds_first   # torch.Size([1, 77, 2048])
                        
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1) # torch.Size([1, 109, 2048]) = torch.Size([1, 77, 2048])  + torch.Size([1, 32, 2048])
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            height=height,
            width=width,
            folder_name=folder_name,
            top=top,
            **kwargs,
        ).images

        return images
