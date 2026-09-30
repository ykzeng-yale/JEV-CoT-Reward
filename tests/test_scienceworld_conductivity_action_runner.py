"""No simulator, gold trajectories, hidden labels, or real outcomes are loaded."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from jev_control.scienceworld_conductivity_policy import ControllerView

spec = importlib.util.spec_from_file_location(
    'conductivity_action_runner',
    Path(__file__).parents[1] / 'scripts/run_scienceworld_conductivity_action_study.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

DESCRIPTION = (
    'Your task is to determine if unknown substance B is electrically conductive. '
    'The unknown substance B is located around the workshop. '
    'First, focus on the unknown substance B. '
    'If it is electrically conductive, place it in the red box. '
    'If it is electrically nonconductive, place it in the green box.')


class Poison:
    def __bool__(self): raise AssertionError('controller accessed hidden metadata')
    def __str__(self): raise AssertionError('controller accessed hidden metadata')
    def __iter__(self): raise AssertionError('controller accessed hidden metadata')
    def __getitem__(self, key): raise AssertionError('controller accessed hidden metadata')
    def get(self, *args): raise AssertionError('controller accessed hidden metadata')


class FakeEnv:
    """Small visible-only house and circuit, with poisonous API metadata."""
    def __init__(self, *, reading='on', missing_kit=False, malformed_task=False,
                 step_failure=None, bad_poststep_actions=False):
        self.reading = reading
        self.missing_kit = missing_kit
        self.malformed_task = malformed_task
        self.step_failure = step_failure
        self.bad_poststep_actions = bad_poststep_actions
        self.loads = []
        self.steps = []
        self.closed = False
        self.variationIdx = None
        self.terminal_calls = []

    @property
    def server(self):
        raise AssertionError('policy accessed private label or object tree')

    def load(self, task, *, variationIdx, simplificationStr, generateGoldPath):
        assert not generateGoldPath
        assert simplificationStr == ''
        self.variationIdx = variationIdx
        self.loads.append(variationIdx)
        self.room = 'kitchen'
        self.opened = False
        self.stopped = False
        self.episode_steps = 0

    def reset(self):
        return 'This room is called the kitchen.', Poison()

    def taskdescription(self):
        return 'invalid visible description' if self.malformed_task else DESCRIPTION

    def get_valid_action_object_combinations(self):
        if self.bad_poststep_actions and self.episode_steps:
            return [Poison()]
        common = ['wait1', 'look around']
        if self.room == 'kitchen': return common + ['go to hallway']
        if self.room == 'hallway':
            return common + (['go to workshop'] if self.opened else ['open door to workshop'])
        refs = [f'{terminal} in {obj}' for obj, terminals in [
            ('battery', ('anode', 'cathode')), ('blue light bulb', ('anode', 'cathode')),
            ('blue wire', ('terminal 1', 'terminal 2')),
            ('green wire', ('terminal 1', 'terminal 2')),
            ('red wire', ('terminal 1', 'terminal 2'))] for terminal in terminals]
        refs.append('unknown substance B')
        connections = [] if self.missing_kit else [
            f'connect {a} to {b}' for a in refs for b in refs if a != b]
        return common + connections + ['focus on unknown substance B',
            'look at blue light bulb', 'move unknown substance B to red box',
            'move unknown substance B to green box']

    def step(self, action):
        assert not self.stopped, 'policy continued after evaluator accessed private state'
        assert action in self.get_valid_action_object_combinations()
        self.steps.append((self.variationIdx, action))
        self.episode_steps += 1
        if self.step_failure is not None:
            raise self.step_failure('fake backend failed after attempted action')
        if action.startswith('go to '):
            self.room = action.removeprefix('go to ')
            response = f'You move to the {self.room}.'
        elif action.startswith('open '):
            self.opened = True
            response = 'The door is now open.'
        elif action == 'look at blue light bulb':
            response = f'A blue light bulb, which is {self.reading}.'
        else:
            response = 'Visible action acknowledgement.'
        return response, Poison(), Poison(), Poison()

    def get_variations_train(self): return range(300)
    def get_variations_dev(self): return range(300, 450)
    def close(self): self.closed = True


@pytest.fixture
def protocol():
    return {'protocol':'fake-only', 'task':'test-visible-circuit', 'random_seed':42,
        'training_integration_ids':[0],
        'splits':{'train':{'start_inclusive':0,'end_exclusive':300},
                  'dev':{'start_inclusive':300,'end_exclusive':450}},
        'policies':['continue_prior','measurement','masked_measurement','random_measurement'],
        'limits':{'episode_action_budget':18,'max_navigation_actions':5,
                  'settling_wait_actions':2,'max_result_bytes':50000000}}


@pytest.fixture(autouse=True)
def terminal_firewall(monkeypatch):
    def terminal(env, target, box, prediction):
        assert target == 'unknown substance B'
        env.stopped = True
        env.terminal_calls.append((target, box, prediction))
        return {'fake_evaluator_after_policy': True}
    monkeypatch.setattr(runner, 'evaluate_terminal', terminal)


def test_navigation_opens_visible_door_and_measurement_uses_no_private_inputs(protocol):
    env = FakeEnv(reading='on')
    row = runner.run_episode(env, protocol, 300, 'measurement', False)
    assert row['failure'] is None
    assert [action for _, action in env.steps[:4]] == [
        'go to hallway','open door to workshop','go to workshop','focus on unknown substance B']
    assert row['prediction'] is True
    assert row['chosen_box'] == 'red box'
    assert row['circuit_actions'] == 6
    assert row['actions'] == 18
    assert row['intrinsic_action_count'] == 14
    assert row['decision_action_index'] == 17
    assert len(env.terminal_calls) == 1
    assert env.steps[-1][1] == 'move unknown substance B to red box'


@pytest.mark.parametrize('policy', ['continue_prior','masked_measurement','random_measurement'])
def test_controls_never_call_reading_decoder_and_randomization_ignores_variation_id(protocol, monkeypatch, policy):
    monkeypatch.setattr(runner, 'indicator_on', lambda *a: pytest.fail('control decoded measurement'))
    results = [runner.run_episode(FakeEnv(reading=reading), protocol, variation, policy, False)
               for reading, variation in [('on', 300), ('off', 449)]]
    assert all(row['failure'] is None for row in results)
    assert results[0]['prediction'] == results[1]['prediction']
    if policy != 'random_measurement': assert results[0]['prediction'] is False
    assert all(row['actions'] == 18 for row in results)
    assert all(row['circuit_actions'] == (0 if policy == 'continue_prior' else 6) for row in results)


def test_view_adapter_carries_exactly_visible_fields():
    env = FakeEnv()
    env.load('x', variationIdx=0, simplificationStr='', generateGoldPath=False)
    observation, _ = env.reset()
    view = runner.view_of(env, observation)
    assert isinstance(view, ControllerView)
    assert set(vars(view)) == {'task_description','observation','legal_actions'}


def test_legally_unavailable_kit_is_retained_as_policy_failure(protocol):
    env = FakeEnv(missing_kit=True)
    row = runner.run_episode(env, protocol, 300, 'measurement', False)
    assert 'complete unused circuit kit' in row['failure']
    assert row['actions'] == 4
    assert len(row['trace']) == 4
    assert row['prediction'] is None and row['decision_action_index'] is None
    assert row['endpoint'] == {'fake_evaluator_after_policy':True}


def test_ambiguous_reading_retains_attempted_circuit_and_no_prediction(protocol):
    env = FakeEnv(reading='on. A blue light bulb, which is off')
    row = runner.run_episode(env, protocol, 300, 'measurement', False)
    assert row['failure'] == 'indicator reading absent or ambiguous'
    assert row['circuit_actions'] == 6
    assert row['reading_sha256'] is not None
    assert row['prediction'] is None
    assert not any(action.startswith('move unknown') for _, action in env.steps)


def install_driver(tmp_path, monkeypatch, env, protocol):
    path = tmp_path/'protocol.json'
    path.write_text(json.dumps(protocol))
    monkeypatch.setattr(runner, 'prepare_runtime', lambda *args: lambda **kwargs:env)
    # Census is tested elsewhere; this isolates the action gate without hidden labels.
    monkeypatch.setattr(runner, 'census', lambda *args:([], {'selected_prior':False}))
    return path, tmp_path/'output'


def test_training_policy_failure_is_saved_and_blocks_all_dev_steps(tmp_path, monkeypatch, protocol):
    env = FakeEnv(missing_kit=True)
    path, output = install_driver(tmp_path, monkeypatch, env, protocol)
    with pytest.raises(RuntimeError, match='no dev action outcomes'):
        runner.run(path, tmp_path/'unused-source', output)
    assert all(variation < 300 for variation, _ in env.steps)
    assert json.loads((output/'training-integration.json').read_text())[0]['failure']
    assert not (output/'episodes.jsonl').exists()
    assert not (output/'result.json').exists()
    assert env.closed


def test_training_setup_failure_is_diagnosed_and_never_reclassified_as_bad_outcome(tmp_path, monkeypatch, protocol):
    env = FakeEnv(malformed_task=True)
    path, output = install_driver(tmp_path, monkeypatch, env, protocol)
    with pytest.raises(ValueError, match='frozen conductivity task contract'):
        runner.run(path, tmp_path/'unused-source', output)
    failure = json.loads((output/'driver-failure.json').read_text())
    assert failure['stage'] == 'training_integration'
    assert failure['variation_id'] == 0 and failure['policy'] == 'measurement'
    assert failure['result_valid'] is False
    assert 'ValueError' in failure['traceback']
    assert env.steps == []
    assert not (output/'result.json').exists()
    assert env.closed


@pytest.mark.parametrize('failure_kind', ['backend_runtime', 'backend_value', 'poststep_view'])
def test_backend_or_poststep_adapter_failure_preserves_attempt_and_blocks_dev(tmp_path, monkeypatch, protocol, failure_kind):
    env = FakeEnv(step_failure={'backend_runtime':RuntimeError,'backend_value':ValueError}.get(failure_kind),
                  bad_poststep_actions=failure_kind == 'poststep_view')
    path, output = install_driver(tmp_path, monkeypatch, env, protocol)
    with pytest.raises(RuntimeError, match='step or visible-view adapter failed'):
        runner.run(path, tmp_path/'unused-source', output)
    records = [json.loads(line) for line in (output/'infrastructure-events.jsonl').read_text().splitlines()]
    assert [record['event'] for record in records] == ['action_started']
    assert records[0]['action'] == 'go to hallway'
    assert records[0]['variation_id'] == 0
    failure = json.loads((output/'driver-failure.json').read_text())
    assert failure['result_valid'] is False
    assert 'direct cause' in failure['traceback']
    assert env.terminal_calls == []
    assert all(variation < 300 for variation, _ in env.steps)
    assert not (output/'result.json').exists()
    assert env.closed


@pytest.mark.parametrize('train_positives,expected', [(150,False),(151,True)])
def test_calibration_is_unchanged_when_only_dev_labels_change(protocol, monkeypatch, train_positives, expected):
    calibrations = []
    for dev_label in (False,True):
        env = FakeEnv()
        monkeypatch.setattr(runner, 'evaluator_label',
            lambda current, dev=dev_label: current.variationIdx < train_positives if current.variationIdx < 300 else dev)
        rows, calibration = runner.census(env, protocol)
        assert len(rows) == 450
        assert env.steps == []
        calibrations.append(calibration)
    assert calibrations[0] == calibrations[1]
    assert calibrations[0]['selected_prior'] is expected
    assert calibrations[0]['train_positive'] == train_positives


@pytest.mark.parametrize('text,room', [
    ('This outside location is called the outside.', 'outside'),
    ('This room is called the kitchen.', 'kitchen'),
    ('You move to the workshop.', 'workshop'),
    ('The door is now open.', None),
])
def test_room_adapter_handles_navigation_responses_without_inventing_location(text,room):
    assert runner.read_room(text) == room
