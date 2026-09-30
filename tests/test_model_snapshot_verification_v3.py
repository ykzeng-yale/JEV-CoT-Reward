"""Content reproducibility including HF LFS/Git identities; no real model required."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.verify_model_snapshot_v3 import (SnapshotVerificationError,parse_checksums,
                                            verify_snapshot,write_receipt)

REVISION='cdbee75f17c01a7cc42f958dc650907174af0554'
SCRIPT=Path(__file__).resolve().parents[1]/'scripts/verify_model_snapshot_v3.py'


def sha(content): return hashlib.sha256(content).hexdigest()


def gitblob(content): return hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()


def snapshot(tmp_path):
    repo=tmp_path/'models--Qwen--Qwen3-4B-Instruct-2507';root=repo/'snapshots'/REVISION
    blobs=repo/'blobs';root.mkdir(parents=True);blobs.mkdir()
    contents={'model.safetensors':b'Fake pinned BF16 weight bytes\x00\x01',
              'config.json':b'{"model_type":"qwen3"}\n',
              'nested/tokenizer.json':b'{"tokenizer":"test"}\n'}
    for name,content in contents.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        if name=='model.safetensors':
            target=blobs/sha(content);target.write_bytes(content);path.symlink_to('../../blobs/'+target.name)
        elif name=='config.json':
            target=blobs/gitblob(content);target.write_bytes(content);path.symlink_to('../../blobs/'+target.name)
        else: path.write_bytes(content)
    manifest=tmp_path/'qualified-model-SHA256SUMS'
    manifest.write_text(''.join(sha(content)+'  '+name+'\n' for name,content in contents.items()))
    return root,manifest,contents


def test_exact_snapshot_verifies_bytes_and_both_hf_content_identity_types(tmp_path):
    root,manifest,contents=snapshot(tmp_path);record=verify_snapshot(root,REVISION,manifest)
    assert record['status']=='PASS_MODEL_SNAPSHOT_CONTENT'
    assert record['file_count']==3 and record['hf_content_addressed_symlinks_verified']==2
    assert record['total_verified_reference_bytes']==sum(map(len,contents.values()))
    files={row['relative_path']:row for row in record['files']}
    assert files['model.safetensors']['content_addressed_blob_type']=='hf_lfs_sha256'
    assert files['config.json']['content_addressed_blob_type']=='hf_git_blob_sha1'
    assert files['config.json']['content_addressed_blob_digest']==gitblob(contents['config.json'])
    assert files['nested/tokenizer.json']['reference_type']=='regular_file'
    assert record['model_loaded'] is False and record['model_generations']==record['downloaded_bytes']==0
    assert record['elapsed_seconds']>=0 and record['process_cpu_seconds']>=0
    assert 'no remote commit-tree attestation' in record['claim_boundary']


@pytest.mark.parametrize('cause,expected',[('corrupt','checksum_mismatch'),('missing','missing_snapshot_file'),
    ('extra','extra_snapshot_file'),('broken','broken_snapshot_symlink'),('revision','snapshot_revision_mismatch')])
def test_corruption_missing_extra_broken_and_wrong_revision_block_promotion(tmp_path,cause,expected):
    root,manifest,_=snapshot(tmp_path)
    if cause=='corrupt': (root/'model.safetensors').resolve().write_bytes(b'corrupt weights')
    elif cause=='missing': (root/'nested/tokenizer.json').unlink()
    elif cause=='extra': (root/'unfrozen.json').write_text('{}')
    elif cause=='broken': (root/'config.json').resolve().unlink()
    elif cause=='revision': root=root.rename(root.with_name('a'*40))
    with pytest.raises(SnapshotVerificationError) as error: verify_snapshot(root,REVISION,manifest)
    assert error.value.code==expected


def test_changed_manifest_cannot_excuse_invalid_lfs_content_address(tmp_path):
    root,manifest,contents=snapshot(tmp_path)
    changed=b'replacement weights'
    (root/'model.safetensors').resolve().write_bytes(changed)
    contents['model.safetensors']=changed
    manifest.write_text(''.join(sha(data)+'  '+name+'\n' for name,data in contents.items()))
    with pytest.raises(SnapshotVerificationError) as error: verify_snapshot(root,REVISION,manifest)
    assert error.value.code=='hf_blob_identity_mismatch'


def test_git_blob_identity_includes_git_object_header_not_raw_sha1(tmp_path):
    root,manifest,contents=snapshot(tmp_path)
    old=(root/'config.json').resolve();wrong=old.with_name(hashlib.sha1(contents['config.json']).hexdigest())
    old.rename(wrong);(root/'config.json').unlink();(root/'config.json').symlink_to('../../blobs/'+wrong.name)
    with pytest.raises(SnapshotVerificationError) as error: verify_snapshot(root,REVISION,manifest)
    assert error.value.code=='hf_blob_identity_mismatch'


@pytest.mark.parametrize('line',[
    '0'*64+'  ../outside', '0'*64+'  /absolute',
    '0'*64+'  config.json\n'+'0'*64+'  ./config.json', 'not a checksum',
])
def test_manifest_scope_and_duplicate_paths_are_not_trusted(tmp_path,line):
    root,manifest,_=snapshot(tmp_path);manifest.write_text(line+'\n')
    with pytest.raises(SnapshotVerificationError): verify_snapshot(root,REVISION,manifest)


def test_sha256sum_dot_relative_and_binary_marker_are_supported(tmp_path):
    root,manifest,contents=snapshot(tmp_path)
    manifest.write_text(''.join(sha(data)+' *./'+name+'\n' for name,data in contents.items()))
    assert set(parse_checksums(manifest))==set(contents)
    assert verify_snapshot(root,REVISION,manifest)['file_count']==3


def test_checksum_and_directory_symlink_cannot_enter_the_snapshot(tmp_path):
    root,manifest,_=snapshot(tmp_path)
    inside=root/'SHA256SUMS';inside.write_bytes(manifest.read_bytes())
    with pytest.raises(SnapshotVerificationError) as error: verify_snapshot(root,REVISION,inside)
    assert error.value.code=='checksum_file_inside_snapshot'
    inside.unlink();(root/'dir_link').symlink_to(root/'nested',target_is_directory=True)
    with pytest.raises(SnapshotVerificationError) as error: verify_snapshot(root,REVISION,manifest)
    assert error.value.code=='directory_symlink'


def test_failed_cli_preserves_receipt_and_never_overwrites_it(tmp_path):
    root,manifest,_=snapshot(tmp_path);(root/'nested/tokenizer.json').unlink();output=tmp_path/'failure.json'
    args=[sys.executable,str(SCRIPT),'--snapshot',str(root),'--revision',REVISION,
          '--checksums',str(manifest),'--output',str(output)]
    first=subprocess.run(args,text=True,capture_output=True)
    assert first.returncode==1
    record=json.loads(output.read_text());assert record['status']=='FAIL_MODEL_SNAPSHOT_CONTENT'
    assert record['error_code']=='missing_snapshot_file'
    preserved=output.read_bytes();second=subprocess.run(args,text=True,capture_output=True)
    assert second.returncode!=0 and output.read_bytes()==preserved
    with pytest.raises(SnapshotVerificationError): write_receipt(output,{'status':'fake'})


def test_earlier_file_mutation_before_end_of_full_snapshot_is_detected(tmp_path,monkeypatch):
    root,manifest,_=snapshot(tmp_path)
    import scripts.verify_model_snapshot_v3 as verifier
    original=verifier.verify_file
    def changing(path,relative,expected):
        record=original(path,relative,expected)
        if relative=='nested/tokenizer.json': (root/'config.json').resolve().write_bytes(b'later mutation')
        return record
    monkeypatch.setattr(verifier,'verify_file',changing)
    with pytest.raises(SnapshotVerificationError) as error: verify_snapshot(root,REVISION,manifest)
    assert error.value.code=='changed_during_verification'
