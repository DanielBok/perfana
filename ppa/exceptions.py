class TimeIndexError(Exception):
    def __init__(self, *args):
        super().__init__("Could not determine time index for series. Please make sure that a date time index is "
                         "specified somewhere.", *args)


class TimeIndexMismatchError(Exception):
    def __init__(self, freq_a: str, freq_b: str, *args):
        err_msg = f"The time frequency for the data sets in comparison are not aligned. One is '{freq_a}' while the " \
            f"other is {freq_b}. Please make sure that they are equivalent for comparison. "

        super().__init__(err_msg, *args)
