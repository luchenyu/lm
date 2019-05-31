#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse, jieba, json, os, requests, time
from flask import Flask, request, make_response, jsonify
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from data import Dataset
from utils import data_utils_py3


parser = argparse.ArgumentParser()
parser.add_argument('port', help="port to serve", type=str)
args = parser.parse_args()

class Client(object):
    def __init__(self, dataset, task_spec, field_mapping, model_name,
                 ip='http://localhost', port=8501):
        self.dataset = dataset
        self.headers = {"content-type": "application/json"}
        self.addr = ip.strip('/')+':'+str(port)+'/v1/models/'+model_name+':predict'
        self.task_spec = task_spec
        self.mapped_index, self.mapped_schema = dataset.task_mapping(
            field_mapping, task_spec)

    def make_request(self, mapped_segments_list):
        req = self.dataset.build_request(
            mapped_segments_list, self.mapped_index, self.mapped_schema, self.task_spec)
        req = json.dumps(req, ensure_ascii=False)
        resp = requests.post(self.addr, data=req, headers=self.headers)
        resp = json.loads(resp.text, encoding='utf-8')
        mapped_segments_list = self.dataset.parse_response(
            resp, self.mapped_index, self.mapped_schema)
        return mapped_segments_list

char_vocab = data_utils_py3.AtomicVocab(
    filename='char_vocab')
dataset = Dataset('./', char_vocab)
task_spec = [
    {
        'name': 'query',
        'type': 'sequence',
        'copy_from': [],
        'target_level': 0,
        'group_id': 0,
    },
    {
        'name': 'response',
        'type': 'sequence',
        'copy_from': [0],
        'target_level': 1,
        'group_id': 1,
    },
]
field_mapping = [0,1]
client = Client(dataset, task_spec, field_mapping, 'chat')
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
register_table = {}



@app.route('/dialog', methods=['POST'])
def reply():
    req = json.loads(request.get_data().decode('utf-8'), encoding='utf-8')
    query = req['inputText']
    mapped_segments_list = [[' '.join(jieba.lcut(query)),'']]
    response = client.make_request(mapped_segments_list)[0][1]
    response = ''.join(response.strip().split(' '))
    res = {'code':'','message':'','score':1.0,'result':response}
    return jsonify(res)


http_server = HTTPServer(WSGIContainer(app))
http_server.listen(args.port)
IOLoop.instance().start()

