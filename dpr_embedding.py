from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from langchain_core.embeddings import Embeddings

class CustomEmbeddings(Embeddings):
    def __init__(self, model_name_or_path, model_kwargs=None):
        if not model_kwargs:
            model_kwargs = {}
        self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_name_or_path)
        self.model = DPRQuestionEncoder.from_pretrained(model_name_or_path, **model_kwargs)
        self.device = model_kwargs["device_map"] if "device_map" in model_kwargs else "cpu" 
    def embed_documents(self, text):
        tmp = self.tokenizer(text, padding=True, return_tensors="pt").to(self.device)
        input_ids, token_type_ids, attention_mask = tmp["input_ids"], tmp["token_type_ids"], tmp["attention_mask"]
        res = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids).pooler_output.detach().cpu().numpy()
        #print(res.shape)
        return res
    def embed_query(self, text: str):
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]

