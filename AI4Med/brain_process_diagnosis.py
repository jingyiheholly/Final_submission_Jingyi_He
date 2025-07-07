import os
import json
import csv
from cli import HuatuoChatbot

# === Initialize Model ===
bot = HuatuoChatbot("/root/autodl-tmp/huatuogpt-vision-7b")

# === Paths ===
annotation_file = '/root/autodl-tmp/VLM_Seminar25_Dataset/nova_brain/annotations.json'
image_base_path = '/root/autodl-tmp/VLM_Seminar25_Dataset/nova_brain/images'
csv_output_file = 'diagnosis_output.csv'

# === Query Template ===
query_template = (
    "You are a radiologist reviewing a brain MRI case.\n\n"
    "Clinical history:\n{clinical_history}\n\n"
    "Imaging findings:\n{captions_text}\n\n"
    "Based on the clinical history and imaging findings, what is the most likely diagnosis?\n"
    "Think through the reasoning carefully if needed, but your final output should contain ONLY the diagnosis as a short phrase "
    "(e.g., 'glioblastoma', 'ischemic stroke', 'brain abscess').\n\n"
    "**Final diagnosis (one line only):**"
)

# === Post-processing function ===
def extract_diagnosis(output, case_id):
    # If output is a list, extract the first item
    if isinstance(output, list):
        output = output[0]

    # Make sure it's a string now
    output = str(output).strip()

    # If already concise
    if len(output.split()) <= 10 and '\n' not in output:
        return output

    # Retry with a clarification prompt
    retry_prompt = (
        f"The previous answer for {case_id} was too long:\n\n"
        f"{output}\n\n"
        "Please now provide ONLY the final diagnosis as a short phrase:"
    )

    try:
        revised = bot.inference(retry_prompt)
        if isinstance(revised, list):
            revised = revised[0]
        revised = str(revised).strip().split('\n')[0]
        return revised
    except Exception as e:
        print(f"Retry failed for {case_id}: {e}")
        return "INVALID"
    


# === Load data ===
with open(annotation_file, 'r') as f:
    data = json.load(f)

# === Open output CSV file ===
with open(csv_output_file, mode='w', newline='', encoding='utf-8') as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow(['case_id', 'diagnosis'])

    # === Iterate through cases ===
    for case_id, case_info in data.items():
        clinical_history = case_info.get("clinical_history", "")
        image_findings = case_info.get("image_findings", {})

        image_paths = []
        captions = []

        for img_name, img_info in image_findings.items():
            full_path = os.path.join(image_base_path, img_name)
            image_paths.append(full_path)
            caption = img_info.get("caption", "")
            captions.append(f"- {caption}")

        captions_text = "\n".join(captions)

        query = query_template.format(
            clinical_history=clinical_history,
            captions_text=captions_text
        )

        try:
            output = bot.inference(query, image_paths)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"⚠️  CUDA OOM on {case_id}, retrying with fewer images...")

                success = False
                for limit in [5, 4, 3, 2, 1]:
                    try:
                        limited_paths = image_paths[:limit]
                        print(f"🔁 Trying with top {limit} images...")
                        output = bot.inference(query, limited_paths)
                        success = True
                        break
                    except RuntimeError as oom_retry_error:
                        if "CUDA out of memory" in str(oom_retry_error):
                            print(f"⚠️  Still OOM with {limit} images.")
                            continue
                        else:
                            print(f"Unexpected error on retry {case_id} with {limit} images: {oom_retry_error}")
                            writer.writerow([case_id, "RETRY_ERROR"])
                            break

                if not success:
                    print(f"❌ All retries failed on {case_id}")
                    writer.writerow([case_id, "OOM_RETRY_FAILED"])
                    continue

            else:
                print(f"Error processing {case_id}: {e}")
                writer.writerow([case_id, "ERROR"])
                continue


        # Handle output and diagnosis extraction
        diagnosis = extract_diagnosis(output, case_id)
        print(f"{case_id} -> {diagnosis}")
        writer.writerow([case_id, diagnosis])

