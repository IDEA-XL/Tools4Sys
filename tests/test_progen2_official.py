from pathlib import Path

import pytest

from progen2.modeling.official import ensure_official_code_dir


def test_ensure_official_code_dir_requires_complete_layout(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        ensure_official_code_dir(str(tmp_path))

    (tmp_path / 'models' / 'progen').mkdir(parents=True)
    (tmp_path / 'models' / '__init__.py').write_text('')
    (tmp_path / 'models' / 'progen' / '__init__.py').write_text('')
    (tmp_path / 'models' / 'progen' / 'configuration_progen.py').write_text('class ProGenConfig: pass')
    (tmp_path / 'models' / 'progen' / 'modeling_progen.py').write_text('class ProGenForCausalLM: pass')
    (tmp_path / 'tokenizer.json').write_text('{}')

    ensure_official_code_dir(str(tmp_path))
