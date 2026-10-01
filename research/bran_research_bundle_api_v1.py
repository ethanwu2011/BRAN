"""Local-only research API. Never print or transmit returned arrays.

Inputs use the exact paired-data clinical schema and existing retinal feature
extractor. This is NOT a raw-image encoder or a clinical decision tool.
"""
from dataclasses import dataclass
import numpy as np
import torch
import bran_research_linear_heads_v1 as heads
from bran_clinical_semantics_v1 import CBC_FIELDS
from run_bran_anchor_ablation_v2 import ELIGIBLE_CONTINUOUS_INDICES
from run_bran_screening_joint_v1 import mask_inputs, PATTERNS

ROUTES=('both','clinical','retinal')
NORMALIZERS=('clinical_median','clinical_iqr','retinal_mean','retinal_scale','age_mean','age_scale')


@dataclass(repr=False)
class ResearchBundle:
    model: object
    normalizers: dict
    names: tuple
    screening_heads: dict
    completion_heads: dict
    experimental: bool


def normalize_inputs(bundle,clinical,observed,retinal,retinal_observed,age):
    c=np.asarray(clinical,float);cm=np.asarray(observed);r=np.asarray(retinal,float);rm=np.asarray(retinal_observed);a=np.asarray(age,float)
    n=len(c)
    if c.shape!=(n,59) or cm.shape!=c.shape or r.shape!=(n,384) or rm.shape!=(n,) or a.shape!=(n,) or cm.dtype!=bool or rm.dtype!=bool:
        raise ValueError('research_input_schema_invalid')
    if (len(bundle.names)!=59 or len(set(bundle.names))!=59 or not set(CBC_FIELDS)<=set(bundle.names)
            or any(bundle.names.index(field)>=48 for field in CBC_FIELDS)):raise ValueError('research_names_invalid')
    eligible=np.zeros(59,bool);eligible[list(ELIGIBLE_CONTINUOUS_INDICES)]=True
    cm=cm&eligible[None]
    if not np.isfinite(c[cm]).all() or not np.isfinite(r[rm]).all() or not np.isfinite(a).all():raise ValueError('research_observed_values_invalid')
    t=bundle.normalizers
    if set(t)!=set(NORMALIZERS):raise ValueError('research_normalizers_invalid')
    for key,width in (('clinical_median',59),('clinical_iqr',59),('retinal_mean',384),('retinal_scale',384)):
        v=np.asarray(t[key])
        if v.shape!=(width,) or not np.isfinite(v).all() or ('scale' in key or 'iqr' in key) and np.any(v<=0):raise ValueError('research_normalizers_invalid')
    if any(np.ndim(t[key])!=0 or not np.isfinite(t[key]) for key in ('age_mean','age_scale')) or t['age_scale']<=0:raise ValueError('research_normalizers_invalid')
    c=np.where(cm,(np.where(cm,c,0.)-t['clinical_median'])/t['clinical_iqr'],0.)
    r=np.where(rm[:,None],(np.where(rm[:,None],r,0.)-t['retinal_mean'])/t['retinal_scale'],0.)
    return c,cm,r,rm,(a-t['age_mean'])/t['age_scale']


def encode(model,c,cm,r,rm,age,route='both',batch_size=256):
    if (route not in ROUTES or type(batch_size)!=int or batch_size<1
            or getattr(getattr(model,'config',None),'state_dim',None)!=192
            or not callable(getattr(model,'encode',None))):raise ValueError('research_route_invalid')
    c=np.asarray(c);cm=np.asarray(cm);r=np.asarray(r);rm=np.asarray(rm);age=np.asarray(age)
    n=len(c)
    if c.shape!=(n,59) or cm.shape!=c.shape or cm.dtype!=bool or r.shape!=(n,384) or rm.shape!=(n,) or rm.dtype!=bool or age.shape!=(n,):raise ValueError('research_encoding_shape_invalid')
    if not np.isfinite(c[cm]).all() or not np.isfinite(r[rm]).all() or not np.isfinite(age).all():raise ValueError('research_encoding_values_invalid')
    if route=='retinal':cm=np.zeros_like(cm)
    if route=='clinical':rm=np.zeros_like(rm)
    c=np.where(cm,c,0.);r=np.where(rm[:,None],r,0.)
    abstain=~(cm.any(1)|rm);z=np.zeros((n,192),float)
    was_training=model.training;model.eval()
    try:
        with torch.no_grad():
            for start in range(0,n,batch_size):
                ix=slice(start,start+batch_size)
                s=model.encode(torch.as_tensor(c[ix],dtype=torch.float32),torch.as_tensor(cm[ix]),
                    torch.as_tensor(r[ix,None],dtype=torch.float32),torch.as_tensor(rm[ix,None]),torch.as_tensor(age[ix],dtype=torch.float32))
                if s.mean.shape!=(len(c[ix]),192) or not torch.isfinite(s.mean).all() or not np.array_equal(s.abstain.numpy(),abstain[ix]):raise ValueError('research_state_invalid')
                z[ix]=s.mean.numpy()
    finally:
        model.train(was_training)
    return np.c_[z,age],abstain


def screen(bundle,clinical,observed,retinal,retinal_observed,age,route='both'):
    x,abstain=encode(bundle.model,*normalize_inputs(bundle,clinical,observed,retinal,retinal_observed,age),route=route)
    return heads.predict_screening(bundle.screening_heads[route],x,abstain)


def complete(bundle,clinical,observed,retinal,retinal_observed,age,pattern='whole_cbc_hidden',field=None):
    if pattern not in PATTERNS or (pattern=='single_target_hidden' and field not in CBC_FIELDS) or (pattern!='single_target_hidden' and field is not None):raise ValueError('research_completion_request_invalid')
    c,cm,r,rm,a=normalize_inputs(bundle,clinical,observed,retinal,retinal_observed,age)
    slots=tuple(bundle.names.index(f) for f in CBC_FIELDS)
    c,cm=mask_inputs(c,cm,slots,pattern,slots[CBC_FIELDS.index(field)] if field else slots[0])
    x,abstain=encode(bundle.model,c,cm,r,rm,a)
    key=pattern+(':'+field if field else '')
    head,calibration=bundle.completion_heads[key]
    return heads.predict_completion(head,calibration,x,abstain)


def serialize_head(head):
    if not isinstance(head,(heads.ScreeningHead,heads.CompletionHead)):raise ValueError('research_head_schema_invalid')
    return {key:getattr(head,key) for key in ('mean','scale','coef','intercept','supported')}


def deserialize_head(value,kind):
    if kind not in (heads.ScreeningHead,heads.CompletionHead) or not isinstance(value,dict) or set(value)!={'mean','scale','coef','intercept','supported'}:raise ValueError('research_head_schema_invalid')
    head=kind(**value);heads._validate_head(head,26 if kind is heads.ScreeningHead else 9,kind);return head
