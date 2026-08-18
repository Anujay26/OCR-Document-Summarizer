from ocr import extract_text


image_path = "test_document.png"

text = extract_text(image_path)

print("\n========== OCR RESULT ==========\n")
print(text)