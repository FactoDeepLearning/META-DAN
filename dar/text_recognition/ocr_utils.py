# Charset / labels conversion
def ctc_remove_successives_identical_ind(ind: list) -> list:
    """
    Remove identical and successive tokens as part of the CTC decoding process.
    :param ind: a list of token indices.
    :return: the list of remaining token indices after process.
    """
    res = []
    for i in ind:
        if res and res[-1] == i:
            continue
        res.append(i)
    return res


def LM_str_to_ind(charset: list[str], sentence: str) -> list:
    """
    Transcribe a text sequence into tokens.
    :param charset: The character set as a list of characters.
    :param sentence: A string sequence to transcribe into token indices.
    """
    return [charset.index(c) for c in sentence]


def LM_ind_to_str(charset: list[str], indices: list[int], oov_symbol:str|None=None) -> str:
    """
    Transcribe a token list into a text sequence.
    :param charset: The character set as a list of characters.
    :param indices: A list of token indices.
    :param oov_symbol: A joker character to use when a token does not match de charset (the CTC blank token for instance)
    """
    if oov_symbol is not None:
        res = []
        for i in indices:
            if i < len(charset):
                res.append(charset[i])
            else:
                res.append(oov_symbol)
    else:
        res = [charset[i] for i in indices]
    return "".join(res)