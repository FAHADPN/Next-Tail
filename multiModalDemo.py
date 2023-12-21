import os
import google.generativeai as genai


GOOGLE_API_KEY = "AIzaSyAZskhE6J9-Df_dlrLfnC3gKWcNYpOKtgM"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

genai.configure(api_key=GOOGLE_API_KEY)
from llama_index.llms import Gemini


from llama_index.multi_modal_llms.gemini import GeminiMultiModal

from llama_index.multi_modal_llms.generic_utils import (
    load_image_urls,
)

image_urls = [
    "https://storage.googleapis.com/generativeai-downloads/data/scene.jpg",
    "https://www.softservedweb.com/_next/image?url=http%3A%2F%2F128.199.22.214%2Fuploads%2Fchatgpt_ea8cd42076_83f28f8b57.png&w=640&q=75",
    "https://www.softservedweb.com/_next/image?url=http%3A%2F%2F128.199.22.214%2Fuploads%2Fsocio_AI_abae5695af_dc9fd39a38.png&w=640&q=75",
    "https://www.softservedweb.com/_next/image?url=http%3A%2F%2F128.199.22.214%2Fuploads%2FAihika_634ed5a8ee.png&w=640&q=75",
    "https://www.softservedweb.com/_next/image?url=http%3A%2F%2F128.199.22.214%2Fuploads%2Fcult_88100bc876_e7509e2d58.png&w=640&q=75",

    # Add yours here!
]

image_documents = load_image_urls(image_urls)

gemini_pro = GeminiMultiModal(model_name="models/gemini-pro-vision")


from PIL import Image
import requests
from io import BytesIO
# import matplotlib.pyplot as plt

img_response = requests.get(image_urls[0])
print(image_urls[0])
img = Image.open(BytesIO(img_response.content))
# plt.imshow(img)

stream_complete_response = gemini_pro.stream_complete(
    prompt="Give me more context for the images",
    image_documents=image_documents,
)

for r in stream_complete_response:
    print(r.text, end="")

