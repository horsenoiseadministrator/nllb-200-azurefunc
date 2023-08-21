import logging

import azure.functions as func



#text = "Gjuhë zyrtare në Republikën e Kosovës janë Gjuha Shqipe dhe Gjuha Serbe."

input_lang = "als_Latn"  #predictions[0][0].replace('__label__', '')


checkpoint = 'facebook/nllb-200-distilled-600M'
# checkpoint = 'facebook/nllb-200-1.3B'
# checkpoint = 'facebook/nllb-200-3.3B'
# checkpoint = 'facebook/nllb-200-distilled-1.3B'


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

target_lang = 'eng_Latn'
translation_pipeline = pipeline('translation', 
                                model=model, 
                                tokenizer=tokenizer, 
                                src_lang=input_lang, 
                                tgt_lang=target_lang, 
                                max_length = 400)



def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    
    text = req.params.get('text')
    
    output = translation_pipeline(text)

    return func.HttpResponse(
             output[0]['translation_text'],
             status_code=200
        )
