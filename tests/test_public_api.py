import src


def test_public_exports_are_available() -> None:
    assert callable(src.reconstruct_nufrost)
    assert callable(src.reconstruct_zhu2015)
    assert callable(src.reconstruct_hants)
    assert callable(src.build_args)
    assert src.RSCube is not None


def test_public_all_contains_expected_symbols() -> None:
    expected = {
        "reconstruct_nufrost",
        "reconstruct_zhu2015",
        "reconstruct_hants",
        "Args",
        "RSCube",
        "build_args",
    }
    assert expected.issubset(set(src.__all__))
