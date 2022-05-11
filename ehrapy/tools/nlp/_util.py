def _list_replace(lst, old: str, new: str):
    """replace list elements (inplace)"""
    i = -1
    try:
        while True:
            i = lst.index(old, i + 1)
            lst[i] = new
    except ValueError:
        pass
