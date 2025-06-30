# ID verification using Fireworks AI

This project built an end-to-end PoV solution to processes images of ID documents (e.g., driver's licenses, passports) using Fireworks AI's vision-language models (VLMs). It classifies the document type, extracts key identity fields, checks for tampering, optionally saves cropped face images and outputs the results in JSON, TXT, or CSV format.

## Features

- Parallel processing of images for faster inference.
- Extraction of identity-related fields (name, DOB, DL number, etc.).
- Document type classification (`passport` or `DL`).
- Forgery detection via VLM/ Multimodal LLM.
- Face bounding box and optional face cropping.
- Output formatting (JSON, TXT, or CSV).

---

## 📂 Input

- A folder containing `.png`, `.jpg`, or `.jpeg` images of ID documents.

---

## Output

- A structured file (JSON, TXT, or CSV) containing the extracted information.
- Optional: a folder with cropped face images from the IDs.

---

## Fields Extracted

| Field        | Description                                  |
|--------------|----------------------------------------------|
| `ID_type`    | Type of ID: `'passport'` or `'DL'`           |
| `dl_number`  | Driver's license number                      |
| `expiry`     | Expiry date of the ID                        |
| `name`       | Full name of the cardholder                  |
| `dob`        | Date of birth                                |
| `address`    | Full address                                 |
| `sex`        | Gender                                       |
| `height`     | Height (e.g., 5'05")                         |
| `weight`     | Weight in lbs                                |
| `hair`       | Hair color                                   |
| `eyes`       | Eye color                                    |
| `altered`    | Whether the document appears altered         |
| `face_bbox`  | `[x, y, width, height]` of the face in image |

---
## VLM Models 

| Model Name                   | Description                                                             |
|-----------------------------|-------------------------------------------------------------------------|
| `qwen2p5-vl-32b-instruct`   | Qwen2.5-VL 32B Multimodal model for vision-language tasks with strong OCR and reasoning | 
| `llama4-maverick-instruct-basic`| Llama 4 leverages a MoE architecture to offer industry-leading performance in text and image understanding   |
| `qwen2p5-vl-7b-instruct`            | Qwen2.5-VL 7B is a multimodal large language model (lightweight)               |
| `deepseek-r1-0528`                | Deepseek R1's this version shows significant improvements in handling complex reasoning tasks                    |
| `deepseek-r1`         | DeepSeek R1 (Fast) is the speed-optimized serverless deployment of DeepSeek-R1                      |


## 🚀 How to Run

- Create [Fireworks AI account](https://fireworks.ai/docs/getting-started/introduction) and follow the instruction [here](https://fireworks.ai/docs/tools-sdks/python-client/the-tutorial) to `Login` and `Install Python SDK` on your local machine.

- Clone this GitHub repo and install `fireworks`, `Pillow`, `argparse`, `json`, `csv`, `os`, `base64`, `time`

- Create an `input` folder storing the input ID documents

- Run the following command to execute the code:

```bash
python id_parser.py \
  --input_dir input_images/ \
  --output_path identity_outputs.json \
  --output_format json \
  --max_workers 6 \
  --model qwen2p5-vl-32b-instruct \
  --face_dir face_crops/
```

---

## 🔧 Command-Line Arguments

| Argument        | Type     | Default                       | Description                                                                 |
|-----------------|----------|-------------------------------|-----------------------------------------------------------------------------|
| `--input_dir`   | `str`    | **(Required)**                | Path to the input folder containing images.                                |
| `--output_path` | `str`    | `identity_outputs.json`       | Output file path.                                                           |
| `--output_format`| `str`   | `json`                        | Output format: `json`, `txt`, or `csv`.                                     |
| `--max_workers` | `int`    | `4`                           | Number of threads used for concurrent image processing.                    |
| `--model`       | `str`    | `qwen2p5-vl-32b-instruct`      | Fireworks model to use for inference.                                      |
| `--face_dir`    | `str`    | `None`                        | If provided, saves cropped face images to this directory.                  |

---

## ⏱️ Performance

The script uses `ThreadPoolExecutor` to encode and process images in parallel. Inference time is printed after processing.

---

## 💡 Notes

- The script is robust against missing or malformed fields (e.g., fills missing entries with `"na"`).
- The `face_bbox` output must be a valid bounding box of length 4; otherwise, cropping is skipped.
- If the model returns `"none (not provided on the DL)"` or similar values, they're normalized to `"na"`.

---

## 📁 Example Output (JSON)

```json
{
  "sample1.png": {
    "ID_type": "DL",
    "dl_number": "A1234567",
    "expiry": "08/31/2024",
    "name": "Jane Doe",
    "dob": "01/01/1990",
    "address": "1234 Main St, City, CA",
    "sex": "F",
    "height": "5-05",
    "weight": "130",
    "hair": "BLK",
    "eyes": "BRN",
    "altered": false,
    "face_bbox": [100, 150, 80, 80]
  }
}
```


---

## Tools Used

- Device: Macbook 14 inch (Apple M2 Pro, macOS: Sequoia.15.5)
- Python 3.9.16
- Pytorch 2.2.0
- `fireworks`
- `Pillow`
- `argparse`
- `json`, `csv`, `os`, `base64`, `time`

### 🔗 Fireworks AI URLs

- [Fireworks API & SDK installation](https://fireworks.ai/docs/tools-sdks/python-client/the-tutorial)
- [Fireworks AI VLM Documentation](https://docs.fireworks.ai/guides/querying-vision-language-models)
- [Fireworks getting started](https://fireworks.ai/docs/getting-started/introduction)

> This project uses FAI's `qwen2p5-vl-32b-instruct` model for document understanding and vision-language reasoning. Please refer to the official documentation for usage guidelines and API references.

