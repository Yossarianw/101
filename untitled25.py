# -*- coding: utf-8 -*-
import subprocess
import os
import shutil
import porespy

# Clone Git Repo
command = 'git clone https://github.com/PMEAL/porespy.git'
subprocess.call(command.split())

# Add porespy/filters directory to the system path
import sys
sys.path.append('porespy/filters')

# Copy and import the required module
src_path = 'porespy/filters/_size_seq_satn.py'
new_folder = 'my_project_folder'
dst_path = os.path.join(new_folder, '_size_seq_satn.py')

if not os.path.exists(new_folder):
    os.mkdir(new_folder)

shutil.copy(src_path, dst_path)

import _size_seq_satn
import importlib
importlib.reload(_size_seq_satn)



# Use the function
_size_seq_satn.size_sequence_saturation(...)

