# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 17:48:35 2018

@author: Ferenc
"""

import time

#time.struct_time(tm_year=2018, tm_mon=10, tm_mday=17, tm_hour=10, tm_min=2, tm_sec=15, tm_wday=2, tm_yday=290, tm_isdst=0)
def get_time_string():
    return time.strftime('%Y%m%d%H%M%S')