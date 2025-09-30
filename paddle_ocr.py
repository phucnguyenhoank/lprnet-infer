# %%
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')

# %%
img_path = "image.png"
results = ocr.predict(img_path)

for res in results:
    res.print()

# %%
