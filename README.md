

# DHumanDiff

Input two face images, the model can generate highly realistic portraits that preserve identity while allowing full customization of poses, outfits, and scenes through text prompts.

---

## 🚀 Features

* 🧑‍🤝‍🧑 **Dual-person synthesis** – seamlessly combine two different individuals in one image
* 🎭 **Prompt-based customization** – control actions, clothing, and backgrounds with natural language
* 👤 **Identity preservation** – maintain facial fidelity across diverse contexts
* 🌍 **Versatile applications** – wedding portraits, couple photos, family/ friends shots, creative photography

---

## 📦 Installation

```bash
git clone https://github.com/yourname/DHumanDiff.git
cd DHumanDiff
pip install -r requirements.txt
```

---

## 📂 Models

Download and place the following pretrained models under the `model/` directory:

| Model                         | Source                                                                                                      | Destination                            |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| Stable Diffusion XL           | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | `model/stable-diffusion-xl-base-1.0/`  |
| CLIP (Text Encoder)           | [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)                       | `model/clip-vit-large-patch14/`        |
| IP-Adapter (Image Encoder)    | [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter/image_encoder)                                       | `model/ipadapter_model/image_encoder/` |
| DHumanDiff pretrained weights | [Pan1111/DHumanDiff](https://huggingface.co/Pan1111/DHumanDiff)                                             | `model/DHumanDiff/`                    |

> ⚠️ Ensure the folder structure matches exactly, otherwise model loading will fail.

---

## ▶️ Run Generation

Example command:

```bash
python src/run_generation.py \
    --image1 data/chenzheyuan.png \
    --image2 data/Anni.png \
    --prompt "A man and a woman are seated on a bench in front of a wall adorned with greenery and flowers. The man is dressed in a black tuxedo. The woman is in a white bridal gown with a veil, holding a bouquet of white flowers." \
    --person_idx1 2 \
    --person_idx2 5
```

### Arguments

| Argument                         | Description                                                 |
| -------------------------------- | ----------------------------------------------------------- |
| `--image1`, `--image2`           | Reference input images                                      |
| `--prompt`                       | Text description guiding the generation                     |
| `--person_idx1`, `--person_idx2` | positions of the described individuals in the prompt text.  |


---

## 🙌 Acknowledgements

* [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
* [IP-Adapter](https://huggingface.co/h94/IP-Adapter)
* [CLIP](https://huggingface.co/openai/clip-vit-large-patch14)

---
