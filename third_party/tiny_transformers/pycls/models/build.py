"""Minimal model registry retained from the audited LG repository.

This vendored copy intentionally builds only the student backbone.  Teacher
loading and distillation live in the surrounding CUB-200 runtime, so importing
the original training wrapper here would pull in unrelated pycls dependencies.
"""

from fvcore.common.registry import Registry

from pycls.core.config import cfg


MODEL = Registry('MODEL')


def build_model():
    return MODEL.get(cfg.MODEL.TYPE)()
