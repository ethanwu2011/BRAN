"""Fail-closed private persistence for the matched unified-refit executor."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import stat

import numpy as np
import torch

ERROR = 'retinal unified refit artifact rejected'
JSON = frozenset(('membership.json', 'inputs_binding.json'))
TORCH = frozenset(('references.pt', *(f'outer{fold}.pt' for fold in range(5)), 'structure.pt'))
NPZ = frozenset(('radii.npz',))
INVENTORY = frozenset((*JSON, *TORCH, *NPZ))
PREPARE = frozenset(JSON)
ORDER = ('membership.json','inputs_binding.json','references.pt','outer0.pt','outer1.pt','outer2.pt','outer3.pt','outer4.pt','structure.pt','radii.npz')


def _fail(): raise ValueError(ERROR) from None
def _req(value):
    if not value: _fail()
def _mode(value): return stat.S_IMODE(value.st_mode)
def _name(name, allowed): _req(type(name) is str and name in allowed and Path(name).name == name)


def _directory(path, *, private=True):
    value = os.lstat(path)
    _req(stat.S_ISDIR(value.st_mode) and not stat.S_ISLNK(value.st_mode))
    if private:
        _req(_mode(value) == 0o700)
    return value
def _regular(path):
    value = os.lstat(path); _req(stat.S_ISREG(value.st_mode) and not stat.S_ISLNK(value.st_mode) and _mode(value) == 0o600 and value.st_nlink == 1); return value
def _digest_fd(fd):
    h = hashlib.sha256(); os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, 1 << 20)
        if not chunk: break
        h.update(chunk)
    return h.hexdigest()
def _hex(value): return type(value) is str and len(value) == 64 and all(x in '0123456789abcdef' for x in value)
def _sha(path):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try: return _digest_fd(fd)
    finally: os.close(fd)


class PrivateArtifacts:
    def __init__(self, path, entry, hashes=None): self._path=Path(path); self._entry=(entry.st_dev,entry.st_ino); self._sealed=False; self._hashes={} if hashes is None else dict(hashes)
    def __repr__(self): return '<PrivateArtifacts private>'
    @classmethod
    def create(cls, root):
        try:
            path=Path(root); _directory(path.parent,private=False); _req(not os.path.lexists(path)); os.mkdir(path,0o700); os.chmod(path,0o700); return cls(path,_directory(path))
        except Exception: _fail()
    @classmethod
    def attach_prepared(cls, root, exact_hashes):
        try:
            authenticate(root, exact_hashes); _req(set(exact_hashes)==set(PREPARE))
            return cls(root, _directory(root), exact_hashes)
        except Exception: _fail()
    def _ready(self):
        _req(not self._sealed); value=_directory(self._path); _req((value.st_dev,value.st_ino)==self._entry)
    def _write(self,name,writer):
        try:
            self._ready(); _req(name == ORDER[len(self._hashes)]); target=self._path/name; _req(not os.path.lexists(target)); temp=self._path/('.'+name+'.'+secrets.token_hex(12)+'.tmp')
            fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'wb') as handle: writer(handle); handle.flush(); os.fsync(handle.fileno())
            os.chmod(temp,0o600); _regular(temp); self._ready(); os.link(temp,target); os.unlink(temp); stored=_regular(target); self._hashes[name]=_sha(target); self._ready()
            dfd=os.open(self._path,os.O_RDONLY); os.fsync(dfd); os.close(dfd)
        except Exception: _fail()
    def write_json(self,name,value):
        _name(name,JSON); self._write(name,lambda h:h.write(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()))
    def write_torch(self,name,value): _name(name,TORCH); self._write(name,lambda h:torch.save(value,h))
    def write_npz(self,name,value):
        _name(name,NPZ); _req(type(value) is dict and value and all(type(k) is str and type(v) is np.ndarray and v.dtype.kind in 'biuf' for k,v in value.items()))
        self._write(name,lambda h:np.savez_compressed(h,**value))
    def manifest(self, expected):
        self._ready(); return authenticate(self._path,expected)
    def seal(self, expected): self.manifest(expected); self._sealed=True; return dict(expected)
    def hashes(self):
        self._ready(); authenticate(self._path,self._hashes); return dict(self._hashes)


def authenticate(root, expected):
    """Authenticate exact private bytes before any deserialization."""
    try:
        path=Path(root); entry=_directory(path); _req(type(expected) is dict and 2 <= len(expected) <= len(ORDER) and set(expected)==set(ORDER[:len(expected)]) and all(_hex(v) for v in expected.values()))
        _req(set(os.listdir(path)) == set(expected))
        result={}
        for name, digest in expected.items():
            _name(name, INVENTORY); artifact=path/name; before=_regular(artifact); fd=os.open(artifact,os.O_RDONLY|os.O_NOFOLLOW)
            try:
                opened=os.fstat(fd); _req((opened.st_dev,opened.st_ino,opened.st_size)==(before.st_dev,before.st_ino,before.st_size) and _digest_fd(fd)==digest)
            finally: os.close(fd)
            after=_regular(artifact); _req((after.st_dev,after.st_ino,after.st_size)==(before.st_dev,before.st_ino,before.st_size))
            result[name]=digest
        _req((entry.st_dev,entry.st_ino)==(_directory(path).st_dev,_directory(path).st_ino)); return result
    except Exception: _fail()


def _opened(root,name,expected):
    authenticate(root,expected); root_entry=_directory(root); path=Path(root)/name; before=_regular(path); fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        opened=os.fstat(fd); _req((opened.st_dev,opened.st_ino,opened.st_size)==(before.st_dev,before.st_ino,before.st_size) and _digest_fd(fd)==expected[name]); return fd,before,(root_entry.st_dev,root_entry.st_ino)
    except Exception: os.close(fd); _fail()
def _bytes(root,name,expected):
    fd,before,root_entry=_opened(root,name,expected)
    try:
        os.lseek(fd,0,os.SEEK_SET); chunks=[]
        while True:
            block=os.read(fd,1<<20)
            if not block: break
            chunks.append(block)
        after=os.fstat(fd); current=_regular(Path(root)/name); current_root=_directory(root)
        _req((after.st_dev,after.st_ino,after.st_size)==(before.st_dev,before.st_ino,before.st_size) and (current.st_dev,current.st_ino,current.st_size)==(before.st_dev,before.st_ino,before.st_size) and (current_root.st_dev,current_root.st_ino)==root_entry and hashlib.sha256(b''.join(chunks)).hexdigest()==expected[name]); authenticate(root,expected)
        return b''.join(chunks)
    finally: os.close(fd)
def load_json(root,name,expected):
    _name(name,JSON)
    try:
        value=json.loads(_bytes(root,name,expected)); authenticate(root,expected); return value
    except Exception: _fail()
def load_torch(root,name,expected):
    _name(name,TORCH)
    try:
        value=torch.load(io.BytesIO(_bytes(root,name,expected)),map_location='cpu',weights_only=False); authenticate(root,expected); return value
    except Exception: _fail()
def load_npz(root,name,expected):
    _name(name,NPZ)
    try:
        loaded=np.load(io.BytesIO(_bytes(root,name,expected)),allow_pickle=False); value={key:loaded[key].copy() for key in loaded.files}; loaded.close()
        _req(all(array.dtype.kind in 'biuf' for array in value.values())); authenticate(root,expected); return value
    except Exception: _fail()
