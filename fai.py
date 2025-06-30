import os
import json
import base64
import time
import csv
import argparse
from fireworks import LLM
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

def encode_image(image_folder, filename):
    path = os.path.join(image_folder, filename)
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return filename, f"data:image/png;base64,{encoded}"

def fill_missing_fields(data):
    for key in data:
        if key not in data or data[key] in [None, "", [], {}, "none (not provided on the DL)", "Not available", "N/A","NULL"]:
            data[key] = "na"
    return data

def process_image(llm, filename, image_data_uri, args):
    try:
        response = llm.chat.completions.create(
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Classify the ID type as 'passport' or 'DL'. Then extract fields: "
                            "DL number, expiry, name, DOB, address, sex, height, weight, hair, eyes. "
                            "Also provide the face crop bounding box. "
                            "Detect if any part of the ID (text, photo, structure) appears generated or altered. "
                            "Output all in JSON, and include a final field named 'altered' with value True or False."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_uri}
                    }
                ]
            }],
            response_format={
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {
                        "ID_type": {"type": "string"},
                        "dl_number": {"type": "string"},
                        "expiry": {"type": "string"},
                        "name": {"type": "string"},
                        "dob": {"type": "string"},
                        "address": {"type": "string"},
                        "sex": {"type": "string"},
                        "height": {"type": "string"},
                        "weight": {"type": "string"},
                        "hair": {"type": "string"},
                        "eyes": {"type": "string"},
                        "altered": {"type": "boolean"},
                        "face_bbox": {"type": "array", "items": {"type": "number"}}
                    },
                    "required": ["ID_type", "name", "dob", "altered"]
                }
            }
        )
        parsed = json.loads(response.choices[0].message.content)

        # remove extra char & missing fields
        parsed = fill_missing_fields(parsed)
        if "height" in parsed:
            parsed["height"] = parsed["height"].replace("\"", "").strip()
        
        # Save face crops
        if args.face_dir:
            face_bbox = parsed.get("face_bbox")
            if face_bbox and len(face_bbox) == 4:
                x, y, w, h = map(int, face_bbox)
                image_path = os.path.join(args.input_dir, filename)
                img = Image.open(image_path)
                face_crop = img.crop((x, y, x + w, y + h))
                face_crop_path = os.path.join(args.face_dir, f"{os.path.splitext(filename)[0]}_face.png")
                face_crop.save(face_crop_path)  

        return filename, parsed
    except Exception as e:
        return filename, {"error": str(e)}

def main(args):
    llm = LLM(model=args.model, deployment_type="serverless")
    output_data = {}

    if args.face_dir:
        os.makedirs(args.face_dir, exist_ok=True)

    image_files = [
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # Encode images
    with ThreadPoolExecutor() as executor:
        encoded_results = list(executor.map(
            lambda f: encode_image(args.input_dir, f),
            image_files
        ))

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_file = {
            executor.submit(process_image, llm, filename, data_uri, args): filename
            for filename, data_uri in encoded_results
        }
        for future in as_completed(future_to_file):
            filename, result = future.result()
            output_data[filename] = result

    elapsed = time.time() - start_time
    print(f"Processed {len(output_data)} images in {elapsed:.2f} seconds.")


    # Save results
    if args.output_format.lower() == "json":
        with open(args.output_path, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"Output saved to {args.output_path}")

    elif args.output_format.lower() == "txt":
        with open(args.output_path, "w") as f:
            for filename, data in output_data.items():
                f.write(f"Filename: {filename}\n")
                for key, value in data.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
        print(f"TXT output saved to {args.output_path}")

    elif args.output_format.lower() == "csv":
        # Flatten fields
        all_fields = [
            "filename", "ID_type", "dl_number", "expiry", "name", "dob",
            "address", "sex", "height", "weight", "hair", "eyes", "face_bbox"
        ]
        with open(args.output_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=all_fields)
            writer.writeheader()
            for filename, data in output_data.items():
                row = {"filename": filename}
                if isinstance(data, dict):
                    row.update({k: str(data.get(k, "")) for k in all_fields if k != "filename"})
                else:
                    row.update({k: "" for k in all_fields if k != "filename"})
                writer.writerow(row)
        print(f"CSV output saved to {args.output_path}")
    else:
        print(f"Output format '{args.output_format}' not supported.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ID document parser with Fireworks VLM")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input images")
    parser.add_argument("--output_path", type=str, default="identity_outputs.json", help="Path to output file")
    parser.add_argument("--max_workers", type=int, default=4, help="Number of threads for parallel inference")
    parser.add_argument("--output_format", type=str, default="json", choices=["json", "txt", "csv"], help="Output format (default: json)")
    parser.add_argument("--model", type=str, default="qwen2p5-vl-32b-instruct", help="Model name (default: qwen2p5-vl-32b-instruct)")
    parser.add_argument("--face_dir", type=str, default=None, help="Directory to save cropped face images")


    args = parser.parse_args()
    main(args)