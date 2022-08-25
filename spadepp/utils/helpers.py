import re

def str2regex(x, match_whole_token=True):
    if not match_whole_token:
        try:
            if x is None:
                return ""
            x = re.sub(r"[A-Z]", "A", x)
            x = re.sub(r"[0-9]", "0", x)
            x = re.sub(r"[a-z]", "a", x)
            return x
        except Exception as e:
            print(e, x)
            return x
    try:
        if x is None:
            return ""
        x = re.sub(r"[A-Z]+", "A", x)
        x = re.sub(r"[0-9]+", "0", x)
        x = re.sub(r"[a-z]+", "a", x)
        x = re.sub(r"Aa", "C", x)
        return x
    except Exception as e:
        print(e, x)
        return x