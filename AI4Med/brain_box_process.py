'''query = (
    "You are a radiologist. Please provide a concise radiology-style description (caption) of the brain MRI image, including: "
    "- MRI sequence type (e.g., T1, T2, FLAIR); "
    "- Anatomic location of abnormality; "
    "- Signal characteristics (e.g., hyperintense, hypointense); "
    "- Possible interpretations (e.g., calcification, hemorrhage)."
)'''
'''query = (
    "You are a radiologist. Please analyze the brain MRI image and locate any abnormal areas, "
    "such as lesions, masses, infarcts, hemorrhages, or calcifications. "
    "For each abnormal region, output only the 2D bounding box coordinates in the format [x_min, y_min, x_max, y_max]. "
    "If there are multiple abnormalities, provide a list of bounding boxes. "
    "If no abnormality is found, return an empty list []."
)'''
'''query = (
    "You are a radiologist reviewing a brain MRI image. "
    "Your task is to locate any abnormal regions such as lesions, hemorrhages, infarcts, or calcifications. "
    "For each abnormal region, output the 2D bounding box in **absolute pixel coordinates** as [x_min, y_min, x_max, y_max].\n\n"
    "Please follow these rules:\n"
    "- Assume the image size is 512×512 unless otherwise specified.\n"
    "- Return a list of bounding boxes if multiple abnormalities are present.\n"
    "- If no abnormality is clearly visible, return an empty list: [].\n"
    "- Be as comprehensive and precise as possible — do not omit subtle or small abnormalities.\n\n"
    "Example of correct bounding box format: [328.5, 443.7, 420.2, 541.8]\n\n"
    "Now, analyze the image and output only the list of bounding box coordinates."
)'''
query = (
    "You are a radiologist reviewing a brain MRI. "
    "Mark all potentially abnormal regions—lesions, hemorrhages, infarcts, calcifications, or any subtle or ambiguous findings.\n\n"
    "Output each as a 2D bounding box in **absolute pixel coordinates**: [x_min, y_min, x_max, y_max].\n\n"
    "Guidelines:\n"
    "- Return a list of boxes if multiple regions exist.\n"
    "- If uncertain, err on the side of inclusion.\n"
    "- If no abnormality is visible, return an empty list: [].\n"
    "- Output **only** the list of bounding boxes.\n\n"
)





from cli import HuatuoChatbot

# Initialize the model
bot = HuatuoChatbot("/root/autodl-tmp/huatuogpt-vision-7b")
import os
import csv
img_folder = '/root/autodl-tmp/VLM_Seminar25_Dataset/nova_brain/images'
img_path = []
csv_file = 'image_boxes_3.csv'
# Iterate over all files in the folder
for filename in os.listdir(img_folder):
    # Optionally filter only image files (e.g., jpg, png, etc.)
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
        full_path = os.path.join(img_folder, filename)
        img_path.append(full_path)
with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['img_path', 'output'])

    for img_path in img_path:
        try:
            output = bot.inference(query, img_path)
            print(f"{img_path} -> {output}")
            writer.writerow([img_path, output])
        except Exception as e:
            print(f"Error processing {img_path}: {e}")