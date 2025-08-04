import os
import json
import csv
from typing import List, Dict, Tuple, Union
from cli import HuatuoChatbot

# === Configuration ===
cfg = OmegaConf.load("../config/config.yaml")
IMG_DIR=cfg.brain.img_dir
ANNOTATION_PATH=cfg.brain.annotations
CSV_OUTPUT_FILE = "diagnosis_output.csv"

QUERY_TEMPLATE = (
    "You are a radiologist reviewing a brain MRI case.\n\n"
    "Clinical history:\n{clinical_history}\n\n"
    "Imaging findings:\n{captions_text}\n\n"
    "Based on the clinical history and imaging findings, what is the most likely diagnosis?\n"
    "Think through the reasoning carefully if needed, but your final output should contain ONLY the diagnosis as a short phrase "
    "(e.g., 'glioblastoma', 'ischemic stroke', 'brain abscess').\n\n"
    "**Final diagnosis (one line only):**"
)


def extract_diagnosis(bot, output, case_id):
    if isinstance(output, list):
        output = output[0]

    output = str(output).strip()

    if len(output.split()) <= 10 and '\n' not in output:
        return output

    retry_prompt = (
        f"The previous answer for {case_id} was too long:\n\n"
        f"{output}\n\n"
        "Please now provide ONLY the final diagnosis as a short phrase:"
    )

    try:
        revised = bot.inference(retry_prompt)
        if isinstance(revised, list):
            revised = revised[0]
        return str(revised).strip().split('\n')[0]
    except Exception as e:
        print(f"Retry failed for {case_id}: {e}")
        return "INVALID"

def safe_infer(bot, query: str, image_paths: List[str], case_id: str) -> Union[str, List[str]]:
    #add try to avoid multiple images causing cuda OFM
    try:
        return bot.inference(query, image_paths)
    except RuntimeError as e:
        if "CUDA out of memory" not in str(e):
            raise e

        print(f"CUDA OOM on {case_id}, retrying with fewer images...")

        for limit in [5, 4, 3, 2, 1]:
            try:
                limited_paths = image_paths[:limit]
                print(f"Retry top {limit} images...")
                return bot.inference(query, limited_paths)
            except RuntimeError as oom_retry_error:
                if "CUDA out of memory" in str(oom_retry_error):
                    print(f"Still OOM with {limit} images.")
                    continue
                else:
                    print(f"Unexpected error on retry for {case_id} with {limit} images: {oom_retry_error}")
                    return "RETRY_ERROR"
        return "OOM_RETRY_FAILED"


def process_case(case_id: str, case_info: Dict, bot) -> Tuple[str, str]:
    clinical_history = case_info.get("clinical_history", "")
    image_findings = case_info.get("image_findings", {})

    image_paths = []
    captions = []

    for img_name, img_info in image_findings.items():
        full_path = os.path.join(IMG_DIR, img_name)
        image_paths.append(full_path)
        caption = img_info.get("caption", "")
        captions.append(f"- {caption}")

    captions_text = "\n".join(captions)

    query = QUERY_TEMPLATE.format(
        clinical_history=clinical_history,
        captions_text=captions_text
    )

    try:
        output = safe_infer(bot, query, image_paths, case_id)
        if output in {"RETRY_ERROR", "OOM_RETRY_FAILED"}:
            return case_id, output
        diagnosis = extract_diagnosis(bot, output, case_id)
        return case_id, diagnosis
    except Exception as e:
        print(f"Error processing {case_id}: {e}")
        return case_id, "ERROR"

def main():
    #add here the actual model path
    bot = HuatuoChatbot("path-to-the-model")

    with open(ANNOTATION_PATH, 'r') as f:
        data = json.load(f)

    with open(CSV_OUTPUT_FILE, mode='w', newline='', encoding='utf-8') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(['case_id', 'diagnosis'])

        for case_id, case_info in data.items():
            case_id, diagnosis = process_case(case_id, case_info, bot)
            print(f"{case_id} -> {diagnosis}")
            writer.writerow([case_id, diagnosis])

    print(f"\nAll results saved to {CSV_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
