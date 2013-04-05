from IPython.parallel import Client
from master.analysis import param_search
import os
import json
import copy
import sys
reload(param_search)

def test(bla):
    """docstring for test"""
    return bla

if __name__ == '__main__':

    data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    all_receptors = json.load(open(os.path.join(data_path, 'all_glomeruli.json')))
    # all_receptors = ["Or22a","Or10a", "Or82a", "Or35a", "Or13a", "Or7a", "Or49b", "Or59b", "Or98a", "Or67c", "Or92a", "Or47a", "Or2a", "Or33b", "Or67b", "Or47b", "Or65a", "Or85b", "Or23a", "Or43a", "Or88a", "Or85a", "Or19a", "Or85f", "Or43b", "Or9a", "ab2B", "Or42b", "Gr21a", "ab3B", "ab5B", "ac2a", "ac2b", "ac1a"]
    all_receptors = ["ac1b","ac4", "ac3a", "Or42a", "Or94b", "Or45a", "Or30a", "Or85c", "Or94a", "Or33a", "Or74a", "Or1a", "Or45b", "Or49a", "Or59a", "Or24a", "Or22c", "Or71a"]
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