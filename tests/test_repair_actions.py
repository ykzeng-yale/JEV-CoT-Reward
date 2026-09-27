from jev_control.repair_actions import ACTIONS, prepare, segment_start

class Tokens:
    def decode(self, ids): return ''.join(map(chr,ids))
    def encode_text(self, text): return list(map(ord,text))


def test_full_segment_repair_preserves_earlier_tokens_and_charges_removed_work():
    b=Tokens();prefix=b.encode_text('established\n\n'+'x'*180+'\n')
    prepared,meta=prepare(b,prefix,'segment_repair')
    assert prepared[:meta['kept_tokens']]==b.encode_text('established\n\n')
    assert meta['removed_tokens']==181
    assert prefix==b.encode_text('established\n\n'+'x'*180+'\n')
    _,legacy=prepare(b,prefix,'suffix_repair')
    assert legacy['removed_tokens']==128


def test_recheck_and_sham_do_not_delete_or_reencode_prefix():
    b=Tokens();prefix=b.encode_text('a\n\nlast\n')
    for action in ('continue','recheck','sham'):
        prepared,meta=prepare(b,prefix,action)
        assert prepared[:len(prefix)]==prefix and meta['removed_tokens']==0
    assert prepare(b,prefix,'continue')[0]==prefix


def test_boundary_ignores_trailing_blank_paragraph():
    b=Tokens()
    assert segment_start(b,b.encode_text('first\n\nlast\n\n'))==7
    assert segment_start(b,b.encode_text('only one\n'))==0
