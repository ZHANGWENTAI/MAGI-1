# Using MAGI-1 in ComfyUI

## Installation

* [Manually download and install ComfyUI](https://github.com/comfyanonymous/ComfyUI?tab=readme-ov-file#manual-install-windows-linux)

* Download this repository into the *ComfyUI/custom\_nodes/MAGI-1* directory and [install the required dependencies](https://github.com/SandAI-org/MAGI-1?tab=readme-ov-file#environment-preparation).

* Download the MAGI-1 model files to your local machine. In the MAGI-1 configuration file—for example, `example/4.5B/4.5B_base_config.json` (if you're using the 4.5B base model)—modify the paths to the model weights so they point to the local files. The following two file paths need to be updated:

    * **load**: Path to the DiT model weights  
    * **vae_pretrained**: Path to the VAE model weights


## Node Functions

After installation, launch ComfyUI from the ComfyUI directory:

```shell
cd ComfyUI
# If you have comfy-cli installed
comfy launch
# Otherwise
python main.py
```

You can find the nodes provided by this repository in the *Add Node - Magi* menu.

> Note: In newer versions of ComfyUI, the nodes are displayed in the left-side NODE LIBRARY panel.

### Load Prompt

Loads a prompt string for text encoding, used in downstream processing.

* **prompt**: User-provided text input. Supports multiline input.

### T5 Text Encoder

Encodes prompt text into conditioning embeddings for video generation.

* **prompt**: The input descriptive text.
* **t5\_pretrained\_path**: Path to the pretrained T5 model weights. Must point to the `ckpt/t5` directory.
* **t5\_device**: The device on which to load and run the T5 model. Options include `"cpu"` or `"cuda:x"` (e.g., `"cuda:0"`).

### Load Image

Loads an image file from the input directory. Supports uploading images via a file selector.

* **image\_path**: Select an image file from ComfyUI’s input folder. Only supported image formats are displayed.

### Image to Video

Core Image-to-Video node. Generates a video sequence from the input image and text embeddings. Also passes the frame rate to the downstream save video node.

* **config\_path**: Path to the JSON configuration file required by the model, e.g., `example/4.5B/4.5B_base_config.json`.
* **image\_path**: Path to the image to be converted into a video (absolute path or relative to the input folder).
* **text\_embeddings**: Embedding and mask produced by the text encoder, used as semantic guidance for video generation.
* **magi\_seed**: Seed for random number generation. Ensures reproducible results. Default is 1234, valid range is 0–100000.
* **video\_size\_h**: Height of the output video (in pixels). Must be a multiple of 16. Default is 720, valid range is 16–14400.
* **video\_size\_w**: Width of the output video (in pixels). Must be a multiple of 16. Default is 720, valid range is 16–14400.
* **num\_frames**: Total number of frames in the generated video. Controls the duration. Must be a multiple of 24. Default is 96, valid range is 24–24000.
* **num\_steps**: Number of diffusion steps. More steps result in higher quality but take more time. Must be a multiple of 4. Default is 64, valid range is 4–240.
* **fps**: Frames per second. Affects playback speed and smoothness. Default is 24, valid range is 1–60.

📌 This node sets several distributed and memory-related environment variables prior to execution to ensure stable operation in multi-GPU environments.

### Save Video

Saves the generated video sequence to a local file.

* **video**: The video tensor to be saved.
* **output\_path**: Full path for saving the video file (only `.mp4` extension is allowed).
* **fps**: Frame rate of the video. Default is 24, range is 1–60.

The video will be encoded and saved to `output_path` with the specified frame rate.

## Workflow Example

This section demonstrates a sample workflow for image-to-video generation. You can load it via the *Load* button in the ComfyUI menu. In newer ComfyUI versions, use *Workflow - Open* from the top-left menu.

The workflow is located in the `comfyui/workflow/` directory, and assets are found in the `example/assets/` directory.

After importing the workflow, **you must manually reassign the corresponding file paths**.

### Image-to-Video Example

Workflow file: `workflow/magi_image_to_video_example.json`