from jev_control.tasks import make_task, verify, shortest_distance


def test_path_ties_and_nonoptimal():
    task={"family":"weighted_path","data":{"n":4,"edges":[[0,1,1],[1,3,2],[0,2,1],[2,3,2],[0,3,9]]}}
    assert verify(task,"FINAL: A->B->D")["success"]
    assert verify(task,"FINAL: A->C->D")["success"]
    assert not verify(task,"FINAL: A->D")["success"]
    assert not verify(task,"FINAL: A->B->C->D")["success"]


def test_expression_exact_multiset_and_no_execution():
    task={"family":"arithmetic_construction","data":{"numbers":[2,2,3],"target":12}}
    assert verify(task,"FINAL: (2+2)*3")["success"]
    assert not verify(task,"FINAL: 12")["success"]
    assert not verify(task,"FINAL: 2**3+2+2")["success"]
    assert not verify(task,"FINAL: __import__('os').system('ls')")["success"]


def test_generated_graph_against_exhaustive_paths():
    for i in range(0,30,2):
        data=make_task(i)["data"]
        def paths(u,cost):
            if u==data["n"]-1: return [cost]
            return [c for a,b,w in data["edges"] if a==u for c in paths(b,cost+w)]
        assert shortest_distance(data)==min(paths(0,0))
