import string

# SETS OF CHARACTERS WHICH CAN BE RECOGNIZED BY CRNN
alphabet28 = string.ascii_lowercase + ' _' # 26 is space, 27 is CTC blank char
alphabet87 = string.ascii_lowercase + string.ascii_uppercase + string.digits + ' +-*.,:!?%&$~/()[]<>"\'@#_'

# DECODES THE OUTPUT OF THE CTC
def decode(chars):
    blank_char = '_'
    new = ''
    last = blank_char
    for c in chars:
        if (last == blank_char or last != c) and c != blank_char:
            new += c
        last = c
    return new
