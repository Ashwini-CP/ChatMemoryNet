
from __future__ import annotations
from flask import Flask, request, jsonify
from .indexer import build_index
from .orchestrator import respond
from .graph_memory import get_graph
from networkx.readwrite import json_graph

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return {'status': 'ok'}

@app.route('/rebuild', methods=['POST'])
def rebuild():
    data = build_index()
    return {'ok': True, 'count': len(data['symptoms'])}

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True)
    user_id = data.get('user_id', 'demo')
    message = data.get('message', '')
    result = respond(user_id, message)
    return jsonify(result)

@app.route('/graph', methods=['GET'])
def graph():
    G = get_graph()
    data = json_graph.node_link_data(G)
    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
