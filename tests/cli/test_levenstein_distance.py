from ehrapy.cli.custom_cli.levenstein_distance import levensthein_dist


def test_levensthein_dist() -> None:
    """
    Test our implemented levensthein distance function for measuring string similarity.
    """
    assert (
        levensthein_dist("horse", "ros") == 3
        and levensthein_dist("", "hello") == 5
        and levensthein_dist("lululul", "") == 7
        and levensthein_dist("intention", "execution") == 5
        and levensthein_dist("", "") == 0
        and levensthein_dist("hello", "") == 5
        and levensthein_dist("cookietemple", "cookiejar") == levensthein_dist("cookiejar", "cookietemple") == 6
        and levensthein_dist("cli", "cliiiiiiiiiii") == 10
        and levensthein_dist("wep", "web") == 1
        and levensthein_dist("mycommand", "mycommand") == 0
    )
