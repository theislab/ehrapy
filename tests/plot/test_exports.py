import inspect

import ehrapy as ep


def test_all_matches_public_api():
    public = {
        name
        for name, obj in vars(ep.pl).items()
        if not name.startswith("_") and (inspect.isfunction(obj) or inspect.isclass(obj))
    }
    assert set(ep.pl.__all__) == public
    assert "hv" not in ep.pl.__all__
