"""Small public procedural tasks for instrumentation, not a new benchmark."""
from __future__ import annotations
import ast
from collections import Counter
from fractions import Fraction
import heapq
import random
import re


def make_task(index, seed=20260927):
    rng = random.Random(seed + index)
    if index % 2 == 0:
        n = 8 + (index % 3)
        edges = {(i, i+1): rng.randint(2, 9) for i in range(n-1)}
        for i in range(n):
            for j in range(i+2, n):
                if rng.random() < .4:
                    edges[i, j] = rng.randint(1, 15)
        data = {"n": n, "edges": [[i,j,w] for (i,j),w in sorted(edges.items())]}
        prompt = (f"Find a minimum-total-weight directed path from A to {chr(65+n-1)}. "
                  "The graph has only the following directed edges with positive weights: " +
                  ", ".join(f"{chr(65+i)}->{chr(65+j)}:{w}" for i,j,w in data["edges"]) +
                  ". Explain intermediate reasoning briefly, checking alternatives. End with exactly FINAL: A->...->" + chr(65+n-1) + ".")
        family = "weighted_path"
    else:
        numbers = [rng.randint(1, 12) for _ in range(6)]
        target = (numbers[0] + numbers[1]) * numbers[2] + numbers[3] * numbers[4] - numbers[5]
        rng.shuffle(numbers)
        data = {"numbers": numbers, "target": target}
        prompt = (f"Use each of these six numbers exactly once: {numbers}. Build an arithmetic expression equal to {target}. "
                  "Allowed operations are +, -, *, / and parentheses; no concatenation, powers, extra constants, or unary minus. "
                  "Explain intermediate reasoning briefly, checking alternatives. End with FINAL: followed by only the expression.")
        family = "arithmetic_construction"
    return {"id": f"{family}-{seed}-{index}", "family": family, "prompt": prompt, "data": data}


def shortest_distance(data):
    best = {0: 0}; queue = [(0, 0)]
    while queue:
        cost, u = heapq.heappop(queue)
        if cost != best[u]: continue
        if u == data["n"]-1: return cost
        for a,b,w in data["edges"]:
            if a == u and cost+w < best.get(b, float("inf")):
                best[b] = cost+w; heapq.heappush(queue, (cost+w,b))
    return float("inf")


def verify(task, text):
    matches = re.findall(r"FINAL:\s*([^\n]+)", text)
    if not matches: return {"success": False, "reason": "missing_final"}
    answer = matches[-1].strip().strip("`$").rstrip(".").strip()
    try:
        if task["family"] == "weighted_path":
            if not re.fullmatch(r"[A-Z](?:\s*->\s*[A-Z])+", answer):
                return {"success": False, "reason": "invalid_path_format"}
            nodes = [ord(x.strip())-65 for x in answer.split("->")]
            data=task["data"]; weights={(a,b):w for a,b,w in data["edges"]}
            valid=nodes[0]==0 and nodes[-1]==data["n"]-1 and len(nodes)==len(set(nodes))
            valid=valid and all((a,b) in weights for a,b in zip(nodes,nodes[1:]))
            cost=sum(weights.get((a,b), 0) for a,b in zip(nodes,nodes[1:]))
            return {"success": bool(valid and cost==shortest_distance(data)), "reason": "checked_path", "path_cost": cost}
        if len(answer)>500: raise ValueError("length")
        tree=ast.parse(answer, mode="eval"); used=[]
        def evaluate(node):
            if isinstance(node, ast.Constant) and type(node.value) is int:
                if not 0<=node.value<=100: raise ValueError("constant")
                used.append(node.value); return Fraction(node.value)
            if not isinstance(node, ast.BinOp): raise ValueError("syntax")
            a,b=evaluate(node.left),evaluate(node.right)
            if isinstance(node.op,ast.Add): return a+b
            if isinstance(node.op,ast.Sub): return a-b
            if isinstance(node.op,ast.Mult): return a*b
            if isinstance(node.op,ast.Div): return a/b
            raise ValueError("operation")
        value=evaluate(tree.body)
        success=value==task["data"]["target"] and Counter(used)==Counter(task["data"]["numbers"])
        return {"success": bool(success), "reason": "checked_expression", "value": str(value)}
    except (ValueError, SyntaxError, ZeroDivisionError, RecursionError):
        return {"success": False, "reason": "invalid_answer"}
