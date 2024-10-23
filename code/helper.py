from pdf2image import convert_from_path
from PIL import Image
import pytesseract


# extract text from pdf image
def read_pdf_image(path):
    pages = convert_from_path(path)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page, lang="chi_tra")

    return text


path = "./reference/finance/5.pdf"
print(read_pdf_image(path))
