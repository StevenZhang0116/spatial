import numpy as np 
import pandas as pd

from helper import *

cell_table = pd.read_feather("./microns_cell_tables/pre_cell_table_microns_mm3.feather")
synapse_table = pd.read_feather("./microns_cell_tables/synapse_table_microns_mm3.feather")

generate_connection(cell_table, synapse_table, "conn.txt")