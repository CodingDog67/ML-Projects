from distutils.core import setup
import py2exe


# DONT USE THIS SHIT, outdated wont support newer python versions, use pyinstaller instead
setup(console=['copy_loin.py'])