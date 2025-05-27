from transformers import AutoMode1ForSeq2SeqLM,AutoTokenizer

model_name = Helsinki-NLP/opus-mt-en-ROMANCE"
save_directory = "./models/opus-mt-en-ROMANCE"

model = AutoMode1ForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"Model saved in:{save_directory}")