from flask import Flask, request, jsonify
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# 파일 경로
questions_file_path = 'pet_hotel_chatbot_ques.txt'
answers_file_path = 'pet_hotel_chatbot_ans.txt'

# 파일 읽기 및 데이터 처리
with open(questions_file_path, 'r', encoding='utf-8') as file:
    questions = file.read().lower().split('\n')

with open(answers_file_path, 'r', encoding='utf-8') as file:
    answers = file.read().lower().split('\n')

# 빈 문자열 제거
questions = [q.strip() for q in questions if q.strip()]
answers = [a.strip() for a in answers if a.strip()]

# 데이터 프레임으로 변환
if len(questions) == len(answers):
    qa_df = pd.DataFrame({'question': questions, 'answer': answers})
else:
    min_len = min(len(questions), len(answers))
    qa_df = pd.DataFrame({'question': questions[:min_len], 'answer': answers[:min_len]})

# 학습 데이터셋 준비
train_df = qa_df.copy()

# 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Mean Pooling 함수 정의
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# 문장 임베딩 생성 함수 정의
def get_sentence_embeddings(sentences):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings

# 트레이닝 데이터셋 질문 임베딩 생성
train_embeddings = get_sentence_embeddings(train_df['question'].tolist())

# 유사한 질문 찾기 함수 정의
def find_most_similar_question(input_question):
    input_embedding = get_sentence_embeddings([input_question])
    similarities = cosine_similarity(input_embedding, train_embeddings)
    most_similar_idx = similarities.argmax()
    return train_df.iloc[most_similar_idx]

# Flask 엔드포인트 정의
@app.route('/get_answer', methods=['POST'])
def get_answer():
    data = request.json
    input_question = data.get('question', '')
    if input_question:
        most_similar_qa = find_most_similar_question(input_question)
        return jsonify({
            'answer': most_similar_qa['answer']
        })
    else:
        return jsonify({'error': 'No question provided'}), 400

if __name__ == '__main__':
    app.run(host='192.168.45.153', port=5000)