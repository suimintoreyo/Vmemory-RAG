from flask import Flask, request, jsonify
import numpy as np
import faiss

app = Flask(__name__)

# FAISSインデックスの初期化
embedding_dim = 1536
index = faiss.IndexFlatL2(embedding_dim)

@app.route('/add', methods=['POST'])
def add_vector():
    data = request.json
    vector = np.array(data['vector'], dtype=np.float32)
    index.add(vector.reshape(1, -1))
    return jsonify({"message": "Vector added!"})

@app.route('/search', methods=['POST'])
def search_vector():
    query_vector = np.array(request.json['vector'], dtype=np.float32).reshape(1, -1)
    distances, indices = index.search(query_vector, k=5)
    return jsonify({"distances": distances.tolist(), "indices": indices.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
