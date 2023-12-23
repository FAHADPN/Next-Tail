import os
import google.generativeai as genai
from PIL import Image
import requests
from io import BytesIO
from llama_index.llms import Gemini
from dotenv import load_dotenv
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.multi_modal_llms.generic_utils import (
    load_image_urls,
)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

def first ():
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


from llama_index.multi_modal_llms.gemini import GeminiMultiModal

from llama_index.multi_modal_llms.generic_utils import (
    load_image_urls,
)

from trulens_eval import TruCustomApp
from trulens_eval import Tru
from trulens_eval.tru_custom_app import instrument
from trulens_eval import Provider
from trulens_eval import Feedback
from trulens_eval import Select
from trulens_eval import TruCustomApp
tru = Tru()
tru.reset_database()
gemini_pro = GeminiMultiModal(model_name="models/gemini-pro-vision")

# create a custom class to instrument
class Gemini:
    @instrument
    def complete(self, prompt, image_documents):
        completion = gemini_pro.complete(
            prompt=prompt,
            image_documents=image_documents,
        )
        return completion

# create a custom gemini feedback provider
class Gemini_Provider(Provider):
    def city_rating(self, image_url) -> float:
        image_documents = load_image_urls([image_url])
        city_score = float(gemini_pro.complete(prompt = "Is the image of a city? Respond with the float likelihood from 0.0 (not city) to 1.0 (city).",
        image_documents=image_documents).text)
        return city_score

gemini_provider = Gemini_Provider()
f_custom_function = Feedback(gemini_provider.city_rating, name = "UI Understandability").on(Select.Record.calls[0].args.image_documents[0].image_url)

def main(url,prompt="Convert this image into HTML and TAILWIND CSS code") :

    image_urls = [
        url
        # Add yours here!
    ]

    image_documents = load_image_urls(image_urls)
    gemini = Gemini()

    gemini_provider.city_rating(image_url=url)
    tru_gemini = TruCustomApp(gemini, app_id = "gemini", feedbacks = [f_custom_function])

    with tru_gemini as recording:
        res = gemini.complete(
        prompt=prompt,
        image_documents=image_documents
        )

        print(res)
        tru.run_dashboard()

if __name__ == "__main__":
    url = "https://next-tail-space.blr1.digitaloceanspaces.com/next-tail/images/website.jpg"
    main(url)



