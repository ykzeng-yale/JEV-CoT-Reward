#!/usr/bin/env python3
"""Verify an exact frozen model snapshot, without importing/loading a model.

Checks the supplied SHA256SUMS file, all snapshot file bytes, and content-addressed
HF symlinks (64-hex LFS SHA256 or 40-hex Git blob SHA1). This is local content
identity, not authenticated remote commit-tree attestation or inference evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time

HEX40=re.compile(r'^[0-9a-f]{40}$')
HEX64=re.compile(r'^[0-9a-f]{64}$')
SUM_LINE=re.compile(r'^([0-9a-fA-F]{64}) ([ *])(.+)$')


class SnapshotVerificationError(ValueError):
    def __init__(self,code,message,relative_path=None):
        super().__init__(message)
        self.code=code
        self.relative_path=relative_path


def require(condition,code,message,relative_path=None):
    if not condition:
        raise SnapshotVerificationError(code,message,relative_path)


def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def parse_checksums(path):
    path=Path(path)
    require(path.is_file(),'missing_checksum_file','Frozen model checksum file is missing.')
    contents=path.read_text(encoding='utf-8')
    result={}
    for line_number,line in enumerate(contents.splitlines(),1):
        if not line.strip(): continue
        match=SUM_LINE.fullmatch(line)
        require(match is not None,'invalid_checksum_manifest',
                f'Invalid SHA256SUMS line {line_number}; snapshot-relative paths required.')
        digest,_,relative=match.groups()
        name=PurePosixPath(relative)
        require(not name.is_absolute() and '..' not in name.parts and name.parts
                and '\\' not in relative and '\x00' not in relative,
                'unsafe_checksum_path','Checksum path must stay within the snapshot.',relative)
        canonical=name.as_posix()
        require(canonical not in result,'duplicate_checksum_path','Duplicate checksum entry.',canonical)
        result[canonical]=digest.lower()
    require(result,'empty_checksum_manifest','Empty checksum manifest cannot qualify a model.')
    return result


def snapshot_files(root):
    files={}
    for directory,dirs,names in os.walk(root,followlinks=False):
        for name in dirs:
            path=Path(directory)/name
            require(not path.is_symlink(),'directory_symlink','Snapshot directory symlinks are unsupported.',
                    path.relative_to(root).as_posix())
        for name in names:
            path=Path(directory)/name
            relative=path.relative_to(root).as_posix()
            try:
                info=path.stat()
            except FileNotFoundError:
                raise SnapshotVerificationError('broken_snapshot_symlink','Missing or broken snapshot target.',relative) from None
            require(stat.S_ISREG(info.st_mode),'non_regular_snapshot_file','Snapshot entry is not a regular file.',relative)
            files[relative]=path
    return files


def metadata_identity(info):
    return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)


def verify_file(path,relative,expected):
    initial_link=os.readlink(path) if path.is_symlink() else None
    initial_target=path.resolve(strict=True)
    with path.open('rb') as stream:
        before=os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode),'non_regular_snapshot_file','Snapshot entry is not a regular file.',relative)
        sha256=hashlib.sha256()
        gitblob=hashlib.sha1()
        gitblob.update(b'blob '+str(before.st_size).encode('ascii')+b'\0')
        read_bytes=0
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            sha256.update(block);gitblob.update(block);read_bytes+=len(block)
        after=os.fstat(stream.fileno())
    require(metadata_identity(before)==metadata_identity(after) and read_bytes==before.st_size,
            'changed_during_verification','Snapshot file changed while hashing.',relative)
    require(metadata_identity(path.stat())==metadata_identity(after)
            and path.resolve(strict=True)==initial_target
            and (os.readlink(path) if path.is_symlink() else None)==initial_link,
            'changed_during_verification','Snapshot reference changed while hashing.',relative)
    actual=sha256.hexdigest()
    require(actual==expected,'checksum_mismatch','Model file differs from frozen SHA256SUMS.',relative)
    blob_type=None;blob_digest=None;blob_verified=None
    blob_name=initial_target.name
    if initial_link is not None and initial_target.parent.name=='blobs':
        if HEX64.fullmatch(blob_name):
            blob_type='hf_lfs_sha256';blob_digest=actual
        elif HEX40.fullmatch(blob_name):
            blob_type='hf_git_blob_sha1';blob_digest=gitblob.hexdigest()
        else:
            raise SnapshotVerificationError('invalid_hf_blob_name','HF blob reference is not content-addressed.',relative)
        require(blob_digest==blob_name,'hf_blob_identity_mismatch',
                'HF symlink target filename does not match its content identity.',relative)
        blob_verified=True
    return {'relative_path':relative,'sha256':actual,'expected_sha256':expected,'bytes':read_bytes,
            'snapshot_reference_path':str(path),'resolved_content_path':str(initial_target),
            'reference_type':'symlink' if initial_link is not None else 'regular_file',
            'declared_symlink_target':initial_link,'content_addressed_blob_name':blob_name if blob_type else None,
            'content_addressed_blob_type':blob_type,'content_addressed_blob_digest':blob_digest,
            'content_addressed_blob_verified':blob_verified},metadata_identity(after)


def verify_snapshot(snapshot,revision,checksums):
    started=time.monotonic();cpu_started=time.process_time()
    root=Path(snapshot).expanduser().absolute()
    require(isinstance(revision,str) and HEX40.fullmatch(revision) is not None,
            'invalid_snapshot_revision','Expected snapshot revision must be a full 40-hex commit ID.')
    require(root.is_dir(),'missing_snapshot','Model snapshot directory is missing.')
    require(root.name==revision and root.resolve(strict=True).name==revision,
            'snapshot_revision_mismatch','Snapshot path does not match the frozen revision.')
    checksum_path=Path(checksums).expanduser().absolute()
    require(checksum_path.is_file(),'missing_checksum_file','Frozen model checksum file is missing.')
    require(not checksum_path.resolve(strict=True).is_relative_to(root.resolve(strict=True)),
            'checksum_file_inside_snapshot','Place SHA256SUMS outside the immutable snapshot.')
    expected=parse_checksums(checksum_path)
    manifest_hash=file_sha256(checksum_path)
    files=snapshot_files(root)
    missing=sorted(set(expected)-set(files));extra=sorted(set(files)-set(expected))
    require(not missing,'missing_snapshot_file','Frozen model files are missing: '+', '.join(missing[:5]))
    require(not extra,'extra_snapshot_file','Snapshot includes unfrozen extra files: '+', '.join(extra[:5]))
    records=[];unique_contents={};verified_metadata={}
    for relative in sorted(expected):
        row,key=verify_file(files[relative],relative,expected[relative])
        records.append(row);unique_contents[key[:2]]=row['bytes'];verified_metadata[relative]=key
    require(set(snapshot_files(root))==set(expected),'changed_during_verification',
            'Snapshot file inventory changed while hashing.')
    for relative,path in files.items():
        require(metadata_identity(path.stat())==verified_metadata[relative],
                'changed_during_verification','Previously hashed file changed before receipt completion.',relative)
    require(file_sha256(checksum_path)==manifest_hash,'changed_checksum_manifest',
            'Frozen checksum manifest changed while hashing.')
    return {'status':'PASS_MODEL_SNAPSHOT_CONTENT','schema':'model_snapshot_content_verification_v3',
            'snapshot_revision':revision,'snapshot_path':str(root),'resolved_snapshot_path':str(root.resolve()),
            'checksum_file_path':str(checksum_path),'checksum_file_sha256':manifest_hash,
            'file_count':len(records),'total_verified_reference_bytes':sum(x['bytes'] for x in records),
            'unique_content_files':len(unique_contents),'unique_content_bytes':sum(unique_contents.values()),
            'hf_content_addressed_symlinks_verified':sum(x['content_addressed_blob_verified'] is True for x in records),
            'non_content_addressed_symlinks':sum(x['reference_type']=='symlink' and x['content_addressed_blob_type'] is None for x in records),
            'files':records,'elapsed_seconds':time.monotonic()-started,
            'process_cpu_seconds':time.process_time()-cpu_started,'model_loaded':False,'model_imports':False,
            'downloaded_bytes':0,'model_generations':0,'hosted_calls':0,'jev_calls':0,
            'claim_boundary':'Exact supplied frozen snapshot content and applicable HF blob identity only; no remote commit-tree attestation, model execution or scientific efficacy.'}


def write_receipt(path,record):
    path=Path(path)
    require(not path.exists(),'existing_receipt','Refuse to overwrite a previous model verification receipt.')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(record,indent=2,sort_keys=True,allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--revision',required=True)
    parser.add_argument('--checksums',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(),'existing_receipt','Refuse to overwrite previous model verification receipt.')
    started=time.monotonic();cpu_started=time.process_time()
    try:
        result=verify_snapshot(args.snapshot,args.revision,args.checksums)
    except (SnapshotVerificationError,OSError,UnicodeError) as exc:
        result={'status':'FAIL_MODEL_SNAPSHOT_CONTENT','schema':'model_snapshot_content_verification_v3',
                'snapshot_revision':args.revision,'snapshot_path':str(args.snapshot),
                'checksum_file_path':str(args.checksums),'error_code':getattr(exc,'code','filesystem_or_encoding_error'),
                'relative_path':getattr(exc,'relative_path',None),'message':str(exc),
                'elapsed_seconds':time.monotonic()-started,'process_cpu_seconds':time.process_time()-cpu_started,
                'model_loaded':False,'model_generations':0,'downloaded_bytes':0}
        write_receipt(args.output,result)
        print(json.dumps({'status':result['status'],'error_code':result['error_code']},sort_keys=True))
        raise SystemExit(1) from None
    result['verifier_sha256']=file_sha256(Path(__file__))
    write_receipt(args.output,result)
    print(json.dumps({k:result[k] for k in ('status','snapshot_revision','file_count','total_verified_reference_bytes',
                                         'checksum_file_sha256','elapsed_seconds')},sort_keys=True))


if __name__=='__main__':main()
