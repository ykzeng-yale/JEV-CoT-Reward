"""Exploratory leave-one-problem-out cheap controller; no test-set or hosted access."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from analyze_action_qualification import analyze
from analyze_screen import estimate,stratified_bootstrap_indices


def fit_predict(train_text,train_y,test_text):
    vec=TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=4000,sublinear_tf=True)
    x=vec.fit_transform(train_text)
    model=Ridge(alpha=10.,solver='lsqr').fit(x,train_y)
    return model.predict(vec.transform(test_text))


def run(root,audit):
    analyze(root,audit)  # refuses stale or incomplete audited data
    cfg=json.loads((root/'manifest.json').read_text())['config'];actions=cfg['actions']
    cps=[json.loads(x) for x in (root/'checkpoints.jsonl').read_text().splitlines()]
    rows=[json.loads(x) for x in (root/'outcomes.jsonl').read_text().splitlines()]
    ids=[c['problem_id'] for c in cps];texts=[json.dumps(c['state'],sort_keys=True) for c in cps]
    y=np.array([[np.mean([r['outcome']['success'] for r in rows if r['problem_id']==pid and r['action']==a]) for a in actions] for pid in ids])
    records=[]
    for i,pid in enumerate(ids):
        train=[j for j in range(len(ids)) if j!=i]
        scores=fit_predict([texts[j] for j in train],y[train],[texts[i]])[0]
        chosen=int(np.argmax(scores));static=int(np.argmax(y[train].mean(axis=0)))
        records.append({'problem_id':pid,'cheap_action':actions[chosen],'training_best_static':actions[static],
            'cheap_success':float(y[i,chosen]),'static_success':float(y[i,static]),'continue_success':float(y[i,actions.index('continue')])})
    ix=stratified_bootstrap_indices([c['task']['family'] for c in cps],10000,20260929)
    return {'status':'exploratory_cross_fitted_development_diagnostic','records':records,
        'cheap_minus_training_best_static':estimate([r['cheap_success']-r['static_success'] for r in records],ix),
        'cheap_minus_continue':estimate([r['cheap_success']-r['continue_success'] for r in records],ix),
        'audit_sha256':hashlib.sha256(audit.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limitations':['Fixed TF-IDF4000/unigram-bigram/Ridge alpha10; no hyperparameter selection. All vocabulary/model/static choices fit without held-out problem.',
        'Overlapping training folds induce dependence; descriptive bootstrap is not a valid independent prospective confidence guarantee.',
        'Small development sample, no fresh deployment or Jev attribution; not a final test or generalization claim.']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Refuse overwrite')
    a.output.write_text(json.dumps(run(a.run,a.audit),indent=2)+'\n')
