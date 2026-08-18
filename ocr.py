import cv2
import pytesseract
import fitz
import numpy as np


pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def preprocess_image(image):

    # Resize
    image = cv2.resize(
        image,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    # Grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Remove noise
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # Threshold
    processed = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return processed


def ocr_image(image):

    processed = preprocess_image(image)

    text = pytesseract.image_to_string(
        processed,
        lang="eng",
        config="--psm 6"
    )

    return text.strip()


def extract_text_from_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Could not read image.")

    return ocr_image(image)


def extract_text_from_pdf(pdf_path):

    document = fitz.open(pdf_path)

    all_text = []

    for page_number, page in enumerate(document):

        # --------------------------------
        # First try normal PDF text
        # --------------------------------

        text = page.get_text().strip()

        if text:

            all_text.append(
                f"\n--- Page {page_number + 1} ---\n"
            )

            all_text.append(text)

        else:

            # --------------------------------
            # Scanned PDF → OCR
            # --------------------------------

            pix = page.get_pixmap(
                matrix=fitz.Matrix(2, 2)
            )

            img = np.frombuffer(
                pix.samples,
                dtype=np.uint8
            )

            img = img.reshape(
                pix.height,
                pix.width,
                pix.n
            )

            if pix.n == 4:

                img = cv2.cvtColor(
                    img,
                    cv2.COLOR_RGBA2BGR
                )

            else:

                img = cv2.cvtColor(
                    img,
                    cv2.COLOR_RGB2BGR
                )

            text = ocr_image(img)

            all_text.append(
                f"\n--- Page {page_number + 1} ---\n"
            )

            all_text.append(text)

    document.close()

    return "\n".join(all_text).strip()