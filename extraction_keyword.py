import torch
import numpy as np
import random
import re
from transformers import BertTokenizer, BertForSequenceClassification
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Prepare stopwords and punctuation
stop_words = set(word.lower() for word in stopwords.words('english'))
punctuations = set(string.punctuation)

# Set random seeds
torch.manual_seed(42)  #Consistent PyTorch behavior
np.random.seed(42) #Consistent NumPy operations
random.seed(42) #Consistent native Python randomness

# Make CuDNN deterministic
torch.backends.cudnn.deterministic = True #Make GPU ops deterministic
torch.backends.cudnn.benchmark = False #Disable performance-based randomness

print("also" in stop_words)  # Should print: True


def clean_text(text):
    tokens = word_tokenize(text)
    cleaned_words = [
        word for word in tokens
        if word.lower() not in stop_words and word not in punctuations
    ]
    return ' '.join(cleaned_words)

def merge_subwords_with_sum(token_score_list):
    merged = []
    buffer = ""
    total_score = 0.0

    for token, score in token_score_list:
        if token.startswith("##"):
            buffer += token[2:]
            total_score += score
        else:
            if buffer:
                merged.append((buffer, total_score))
            buffer = token
            total_score = score

    if buffer:
        merged.append((buffer, total_score))

    return merged
    
def compute_percentiles(data):
    """
    data: List of tuples (token, value)
    Returns: List of tuples (token, value, percentile)
    """
    N = len(data)
    values = [v for _, v in data]
    
    result = []
    for i, (token, value) in enumerate(data):
        # Count how many values are strictly less than the current one
        n = sum(1 for v in values if v < value)
        percentile = round((n / (N - 1)) * 100, 2)
        result.append((token, percentile))
    
    return result
    
def get_prediction_score(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():                        #Runs the model in evaluation mode (torch.no_grad() to save memory).
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs[0][1].item()  # Assuming class 1 is the target

def get_important_keywords_by_score_drop(text, extractor):
    original_score = get_prediction_score(extractor.model, extractor.tokenizer, text)
    print("Original prediction score:", original_score)

    keywords = extractor.extract_keywords(text)
    print("Initial extracted keywords:", keywords)

    drop_data = []
    
    for word in keywords:
        # Mask this single word only
        masked_text = re.sub(
            r'\b' + re.escape(word) + r'\b', '[MASK]', text, flags=re.IGNORECASE
        )
        masked_score = get_prediction_score(extractor.model, extractor.tokenizer, masked_text)
        drop = original_score - masked_score
        drop_data.append((word, drop))
        print(f"Masked '{word}': drop = {drop:.4f}")

    if not drop_data:
        return []

    # Find max drop
    max_drop = max(drop_data, key=lambda x: x[1])[1]
    threshold = 0.8 * max_drop
    print(f"Max drop = {max_drop:.4f}, Threshold (80%) = {threshold:.4f}")

    # Select important keywords
    important_keywords = [word for word, drop in drop_data if drop >= threshold]
    return important_keywords


    
class GradientKeywordExtractor:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.model.eval()  # Set to eval mode

    def extract_keywords(self, text):
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Get embeddings and enable gradient tracking
        input_embeddings = self.model.bert.embeddings(input_ids)
        input_embeddings.retain_grad()
        input_embeddings.requires_grad_(True)

        # Forward pass using custom embeddings
        outputs = self.model(inputs_embeds=input_embeddings, attention_mask=attention_mask)
        logits = outputs.logits
        pred_label = torch.argmax(logits, dim=-1)

        # Backward pass: compute gradient w.r.t. predicted class
        self.model.zero_grad()
        logits[0][pred_label].backward()

        # Get gradient and compute norm for each token
        gradients = input_embeddings.grad[0]  # shape: (seq_len, hidden_dim)
        grad_norms = torch.norm(gradients, dim=1).detach().cpu().numpy()

        # Convert token IDs to tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        # Zip tokens with gradient norms
        token_scores = list(zip(tokens, grad_norms))
        print("token_scores", token_scores)
        merged_keywords = merge_subwords_with_sum(token_scores)
        print("merged_keywords", merged_keywords)
        cleaned = []
        for token, score in merged_keywords:
            if token in ['[CLS]', '[SEP]', '[PAD]', '.', '?'] or re.match(r'^\W+$', token):
                continue
            if token.startswith("##"):
                continue
            cleaned.append((token, score))

        if not cleaned:
            return []
            
        merged_keywords = compute_percentiles(cleaned)
        print("percentile", merged_keywords)
       
        sorted_items = sorted(merged_keywords, key=lambda item: item[1], reverse=True)
        print("sorted_items", sorted_items)
        
   
        threshold = 50
        keywords = [term for term, score in sorted_items if score >= threshold]

        return keywords
        
    



extractor = GradientKeywordExtractor()
text = "ingredient works on skin condition"
print(text)
text = clean_text(text)
print(text)
important_keywords = get_important_keywords_by_score_drop(text, extractor)
print("Important Keywords based on Score Drop:", important_keywords)

# keywords = extractor.extract_keywords(text)
# print("Extracted Keywords:", keywords)

# words_to_mask = keywords


# masked_sentence = re.sub(
    # r'\b(' + '|'.join(re.escape(word) for word in words_to_mask) + r')\b',
    # '[MASK]',
    # text,
    # flags=re.IGNORECASE
# )

# print(masked_sentence)
# Run on masked sentence
# keywords = extractor.extract_keywords(masked_sentence)
# print("Extracted Keywords:", keywords)
