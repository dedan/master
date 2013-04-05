from IPython.parallel import Client
from master.analysis import param_search
import os
import json
import copy
import sys

def test(bla):
    """docstring for test"""
    return bla

if __name__ == '__main__':

    data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    all_receptors = json.load(open(os.path.join(data_path, 'all_glomeruli.json')))
    all_receptors = ["Or22a", "Or10a"]

    sc = json.load(open(sys.argv[1]))

    configs = []
    for receptor in all_receptors:
        tmp_config = copy.deepcopy(sc)
        tmp_config['glomeruli'] = [receptor]
        configs.append(tmp_config)

    rc = Client()
    dview = rc.direct_view()
    dview.block = True
    dview.execute('from master.analysis import param_search')
    dview.execute("reload(param_search)")
    dview.block = False
    lview = rc.load_balanced_view()
    out = lview.map(param_search.paramsearch, configs)
    # out = lview.map(test, configs)

    for i in out:
        print i