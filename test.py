from sentence_transformers.models.PhoBERT import PhoBERT

model = PhoBERT.load('/workspace/PhoBERT')





sentences = ['Hôm nay là thứ tư, ngày 13 tháng 5.', "Tôi là người Việt Nam.", "Mây lấp ló sau từng tán lá, chợt ôm trọn lấy ông mặt trời."]


sentence_embeddings = model.get_sentence_features(model.tokenize(sentences[0]), 10)
print(sentence_embeddings)

