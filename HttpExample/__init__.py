import logging

import azure.functions as func



input_lang = "als_Latn" 

# version of nllb you wish to use, more params is more powerful
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

  # Add to URL param ?text=.... to input text to the API
  
    text = req.params.get('text')
    
    output = translation_pipeline(text)

    return func.HttpResponse(
             output[0]['translation_text'],
             status_code=200
        )
