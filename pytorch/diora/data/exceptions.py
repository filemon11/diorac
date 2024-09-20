class BracketFormatError(ValueError):
    def __init__(self):
        super().__init__("bracket argument must contain two symbols (one opening and one closing bracket)")

def check_bracket_format(brackets : str) -> None:
    if (len(brackets) != 2 
        or not isinstance(brackets[0], str) 
        or not isinstance(brackets[1], str)):
        
        raise BracketFormatError