# Using MAGI-1 in ComfyUI

## Installation Guide

* [Download and install ComfyUI manually](https://github.com/comfyanonymous/ComfyUI?tab=readme-ov-file#manual-install-windows-linux)

* Clone this repository into the path: *ComfyUI/custom\_nodes/MAGI-1*, and [install the required dependencies](https://github.com/SandAI-org/MAGI-1?tab=readme-ov-file#environment-preparation).

  > \[!NOTE]
  >
  > To make ComfyUI recognize custom nodes, you must move the `comfyui/__init__.py` file to the root directory of MAGI-1.

* Download the MAGI-1 model files to your local system. Modify the model weight paths in the MAGI-1 config file, e.g., `example/4.5B/4.5B_base_config.json` (for using the 4.5B base model). You need to update two paths:

  * **load**: Path to the DiT model weights.
  * **vae\_pretrained**: Path to the VAE model weights.

## Node Features

After installation, navigate to the ComfyUI directory and launch ComfyUI:

```shell
cd ComfyUI
# If you have comfy-cli installed
comfy launch
# Otherwise
python main.py
```

You can find the nodes provided by this repository under *Add Node - Magi* in ComfyUI. In newer versions, you can also find them in the NODE LIBRARY panel on the left.

### Load Prompt

Loads a prompt text from the input to be used for later text encoding.

* **prompt**: User input text, supports multiline input.

### T5 Text Encoder

Encodes prompt text into text features (Conditioning Embedding) for video generation.

* **prompt**: Input descriptive text.
* **t5\_pretrained\_path**: Absolute path to the T5 model weights, pointing to the pre-trained model in the `ckpt/t5` directory.
* **t5\_device**: Specifies the device on which to load and run the T5 model, options are `"cpu"` or `"cuda:x"` (e.g., `"cuda:0"`).

### Load Image

Loads an image file from the input directory. File picker supports image format filtering.

* **image\_path**: Select an image file from ComfyUI's input folder. Only supported image types will be shown.

### Process with MAGI

The core node for tasks like text-to-video, image-to-video, or video continuation. Generates the video sequence and passes the frame rate to the save node.

* **task\_mode**: Select from *Text-to-Video*, *Image-to-Video*, or *Video Continuation*.
* **config\_path**: Absolute path to the JSON config file required by the model.

  > \[!NOTE]
  >
  > All paths in the config file must be absolute paths.
* **image\_path**: Absolute path to the image or video to be used for generation.
* **text\_embeddings**: Embeddings and mask from the text encoder, used to guide video generation semantically.
* **magi\_seed**: Random seed for reproducibility. The same seed produces the same video. Default is 1234. Valid range: 0–100000.
* **video\_size\_h**: Height (pixels) of the generated video. Default is 720. Larger sizes may slow performance or cause memory overflow.
* **video\_size\_w**: Width (pixels) of the generated video. Default is 720. Be cautious with large values.
* **num\_frames**: Total number of frames in the generated video. Controls video duration. Default: 96. Range: 24–24000.
* **num\_steps**: Number of diffusion sampling steps. More steps produce higher quality but take more time. Default: 64. Range: 4–240.
* **fps**: Frames per second. Controls playback speed and smoothness. Default is 24. Range: 1–60.

> \[!NOTE]
>
> This node sets a series of distributed and memory-related environment variables before running.

### Save Video

Saves the generated video sequence to a local file.

* **video**: The video tensor to be saved (a `torch.Tensor`).
* **output\_path**: Absolute path to save the video. Must use the `.mp4` extension.
* **fps**: Frame rate for the video. Default is 24. Range: 1–60.

The video will be encoded using the specified FPS and written to `output_path`.

## Workflow Examples

This section demonstrates example workflows for image-to-video generation. You can import these workflows using the *Load* button in the menu. In newer versions of ComfyUI, go to *Workflow - Open* from the top-left menu.

Workflows are located in the `comfyui/workflow/` directory, and assets are in the `example/assets/` directory.

After importing a workflow, **you must manually reassign the correct file paths**.

### Text-to-Video

Workflow file: `workflow/magi_text_to_video_example.json`

### Image-to-Video

Workflow file: `workflow/magi_image_to_video_example.json`

### Video Continuation

Workflow file: `workflow/magi_video_continuation_example.json`
