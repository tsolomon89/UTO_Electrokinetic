import py_compile

def test_compiles():
    py_compile.compile('Version512_Working.py', doraise=True)
