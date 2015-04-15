__author__ = 'ilap'

from bioLibrary import *

### MaAIN
# DONE string = "am$al"
# DONE print reconstructTextFromBWT(string)

text = "smnpbnnaaaaa$a"
print reconstructTextFromBWT(text)
patterns = "ana".split()

result = []
for pattern in patterns:
    result.append(str (patternMatchInBWT (text, pattern)))

print ' '.join (result)



