import nplab


df = nplab.current_datafile()


def make_gr():
    df.require_group('thing')
