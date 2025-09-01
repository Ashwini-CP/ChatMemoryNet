
from __future__ import annotations
from typing import Dict, Any, List

from langchain_core.runnables import RunnableLambda, RunnableParallel
from .nlp import extract_entities, is_tanglish
from .indexer import search_symptom
from .graph_memory import get_graph, save_graph, add_user_if_missing, upsert_thread, add_message, add_symptom_solution

def _build_anchors(user_text: str, solution_text: str) -> List[Dict[str, Any]]:
    ents = extract_entities(user_text)
    anchors = [{'key': 'ENT::' + e['label'], 'value': e['text'], 'weight': 1.0} for e in ents]
    anchors.append({'key': 'has_solution', 'value': solution_text, 'weight': 1.0})
    return anchors

def respond(user_id: str, message: str) -> Dict[str, Any]:
    # Chain 1: retrieve solution from symptom text
    def retrieve(msg: str):
        idx, sol, score = search_symptom(msg, top_k=3)
        return {'solution': sol, 'score': score}

    # Chain 2: style response (Tanglish echo if message looks tanglish)
    def stylize(inp):
        sol = inp['solution']
        if is_tanglish(message):
            return sol  # assume dataset solution in tanglish
        return sol

    pipeline = RunnableParallel(retriever=RunnableLambda(retrieve)) | RunnableLambda(lambda d: {'text': stylize(d['retriever']), **d['retriever']})
    out = pipeline.invoke(message)

    # Update graph memory
    G = get_graph()
    add_user_if_missing(G, user_id)
    theme = 'health'
    thread_id = upsert_thread(G, user_id, theme)
    anchors = _build_anchors(message, out['text'])
    add_message(G, user_id, thread_id, role='user', text=message, anchors=[])
    add_message(G, user_id, thread_id, role='assistant', text=out['text'], anchors=anchors)
    add_symptom_solution(G, message, out['text'], user_id, thread_id)
    save_graph(G)

    return {'reply': out['text'], 'score': out['score'], 'thread_id': thread_id}
