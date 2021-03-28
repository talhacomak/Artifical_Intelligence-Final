This is a group project of Artifical Intelligence course.

about project:
video link: https://drive.google.com/file/d/1KjM74c_2LvWHl2namujDJYp4JZWCheA8/view

if virtual environment does not exist:
> pip install --upgrade pip
> python3 -m pip install --user virtualenv

download and extract code.
> python -m venv venv
> source venv/bin/activate
> pip install --upgrade pip
> cd gym-duckietown
> pip3 install -e .
> pip3 install imutils
> pip3 install opencv-contrib-python
> cp xyz.yaml ../venv/lib/python3.6/site-packages/duckietown_world/data/gd1/maps/
> gedit ../venv/lib/python3.6/site-packages/duckietown_world/world_duckietown/old_map_format.py

change:

from typing import Dict, List, NewType, TypedDict, Union

to:

from typing import Dict, List, NewType, Union
from typing_extensions import TypedDict

save and close.

> python main.py --map-name xyz


please contact kacmazh17@itu.edu.tr if faced with any troubles.
